import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader 


class Generator(nn.Module):

    def __init__(self, latent_dim=100, num_classes=43):

        super().__init__()

        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(

            nn.ConvTranspose2d(latent_dim + num_classes, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512,256,4,2,1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256,128,4,2,1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128,64,4,2,1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64,3,4,2,1,bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):

        label = self.label_emb(labels)
        label = label.unsqueeze(2).unsqueeze(3)

        x = torch.cat((noise, label), dim=1)

        return self.model(x)



class Discriminator(nn.Module):

    def __init__(self, num_classes=43):

        super().__init__()

        self.label_emb = nn.Embedding(num_classes, 64*64)

        self.model = nn.Sequential(

            nn.Conv2d(4,64,4,2,1,bias=False),
            nn.LeakyReLU(0.2,True),

            nn.Conv2d(64,128,4,2,1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,True),

            nn.Conv2d(128,256,4,2,1,bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,True),

            nn.Conv2d(256,512,4,2,1,bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,True),

            nn.Conv2d(512,1,4,1,0,bias=False)
        )

    def forward(self, img, labels):

        label = self.label_emb(labels)
        label = label.view(-1,1,64,64)

        x = torch.cat((img,label),dim=1)

        return self.model(x).view(-1)


class Encoder(nn.Module):

    def __init__(self, latent_dim=100, num_classes=43):

        super().__init__()

        self.label_emb = nn.Embedding(num_classes, 64 * 64)

        self.features = nn.Sequential(

            nn.Conv2d(4, 64, 4, 2, 1, bias=False),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)

    def forward(self, img, labels):

        label = self.label_emb(labels)
        label = label.view(-1, 1, 64, 64)

        x = torch.cat((img, label), dim=1)

        x = self.features(x)

        x = torch.flatten(x, start_dim=1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


def reparameterize(mu, logvar):

    std = torch.exp(0.5 * logvar)

    eps = torch.randn_like(std)

    z = mu + eps * std

    return z


def weights_init(m):

    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data,0.0,0.02)

    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.constant_(m.bias.data,0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


latent_dim = 100
num_classes = 43

generator = Generator(latent_dim, num_classes).to(device)
discriminator = Discriminator(num_classes).to(device)

generator.apply(weights_init)
discriminator.apply(weights_init)


loss = nn.BCEWithLogitsLoss()

gen_optim = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
dis_optim = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

num_epoches = 1000


class GTSRB_Dataset(Dataset):

    def __init__(self, r_dir, csv_file, transform=None):

        self.r_dir = r_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        path = os.path.join(self.r_dir, self.data.iloc[index]["Path"])
        image = Image.open(path).convert("RGB")
        label = int(self.data.iloc[index]["ClassId"])

        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor(), transforms.Normalize((0.5,0.5 ,0.5), (0.5, 0.5, 0.5))])

r_dir = r"path to dataset"
csv_file = r"path to Train.csv"

train_data = GTSRB_Dataset(r_dir=r_dir, csv_file=csv_file, transform=transform)

train_load = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)

encoder = Encoder(latent_dim, num_classes).to(device)
encoder.apply(weights_init)

enc_optim = optim.Adam(encoder.parameters(), lr=0.0002, betas=(0.5,0.999))


loss_recon = nn.L1Loss()

lambda_adv = 1.0
lambda_rec = 10.0
lambda_kl = 0.001


for epoch in range(num_epoches):

    for real_imgs, labels in train_load:

        real_imgs = real_imgs.to(device)
        labels = labels.to(device)

        batch = real_imgs.size(0)

        real = torch.ones(batch, device=device)
        syn = torch.zeros(batch, device=device)


        # -------------- Discriminator -------------- 


        dis_optim.zero_grad()

        output_real = discriminator(real_imgs, labels)
        loss_real = loss(output_real, real)

        # noise = torch.randn(batch, latent_dim, 1, 1, device=device)
        # syn_imgs = generator(noise, labels)

        with torch.no_grad():
            noise = torch.randn(batch, latent_dim, 1, 1, device=device)
            syn_imgs = generator(noise, labels)

        output_syn = discriminator(syn_imgs.detach(), labels)
        loss_syn = loss(output_syn, syn)

        d_loss = loss_real + loss_syn
        d_loss.backward()
        dis_optim.step()


        # ---------------- Generator ----------------


        enc_optim.zero_grad()
        gen_optim.zero_grad()

        mu, logvar = encoder(real_imgs, labels)
        z = reparameterize(mu, logvar)
        z = z.unsqueeze(2).unsqueeze(3)

        recon_imgs = generator(z, labels)
        recon_loss = loss_recon(recon_imgs, real_imgs)

        kl_loss = -0.5*torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        noise = torch.randn(batch, latent_dim, 1, 1, device=device)
        syn_imgs = generator(noise, labels)

        syn_pred = discriminator(syn_imgs, labels)

        adv_loss = loss(syn_pred, real)


        g_loss = lambda_adv*adv_loss + lambda_kl*kl_loss + lambda_rec*recon_loss
        g_loss.backward()
        enc_optim.step()
        gen_optim.step()

    print(f"Epoch {epoch+1}/{num_epoches} | D={d_loss.item():.3f} | G={g_loss.item():.3f} | KL={kl_loss.item():.3f} | ADV={adv_loss.item():.3f} | REC={recon_loss.item():.3f}")

torch.save({
    "epoch": epoch,
    "generator_state_dict": generator.state_dict(),
    "discriminator_state_dict": discriminator.state_dict(),
    "encoder_state_dict": encoder.state_dict(),
    "generator_optimizer": gen_optim.state_dict(),
    "discriminator_optimizer": dis_optim.state_dict(),
    "encoder_optimizer": enc_optim.state_dict(),
    "latent_dim": latent_dim,
    "num_classes": num_classes,
}, "cgan_vae.pth")