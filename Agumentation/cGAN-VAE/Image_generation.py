import torch
import torchvision
from model_training import Generator
from torchvision.utils import save_image
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = torch.load("cgan_vae.pth", map_location=device)


latent_dim = save_path["latent_dim"]
num_classes = save_path["num_classes"]

generator = Generator(latent_dim, num_classes).to(device)

generator.load_state_dict(save_path["generator_state_dict"])

# ----------------- Generation ----------------- 

generator.eval()

class_id = 1      

save_dir = f"cgan_vae_class_{class_id}"
os.makedirs(save_dir, exist_ok=True)

for i in range(20):

    noise = torch.randn(1, 100, 1, 1).to(device)
    label = torch.tensor([class_id], dtype=torch.long).to(device)

    with torch.no_grad():
        syn = generator(noise, label)

    torchvision.utils.save_image(syn, os.path.join(save_dir, f"class{class_id}_{i+1}.png"), normalize=True)

print("Images generated.")