import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from PIL import Image
from matplotlib.patches import Rectangle

# Data Loading And Analysis

meta_csv = pd.read_csv("Inputs/Meta.csv")
train_csv = pd.read_csv("Inputs/Train.csv")
test_csv = pd.read_csv("Inputs/Test.csv")
print(meta_csv.head())
print(meta_csv.shape)
print(train_csv.head())
print(train_csv.shape)
print(test_csv.head())
print(test_csv.shape)

# Shape Distribution
plt.figure()
plt.bar(pd.unique(meta_csv["ShapeId"]),meta_csv['ShapeId'].value_counts())
plt.xticks(
    [0,1,2,3,4],
    ['Triangle', 'Circle', 'Diamond' , 'InvTriangle', 'Octagon'], 
    rotation=45, 
    fontsize=10)
plt.title("Shape Distribution over Class")
plt.tight_layout()
plt.savefig("Outputs/Shape_Distribution_Over_Class.png")
plt.close()

# Color Distribution
plt.figure()
plt.bar(pd.unique(meta_csv["ColorId"]),meta_csv['ColorId'].value_counts())
plt.xticks(
    [0,1,2,3],
    ['Red-White', 'White-Yellow', 'Blue', 'White-Black'], 
    rotation=45, 
    fontsize=10)
plt.title("Color Distribution over Class")
plt.tight_layout()
plt.savefig("Outputs/Color_Distribution_Over_Class.png")
plt.close()

# Image Count Verification
train_path = Path("Inputs/Train")
test_path = Path("Inputs/Test")

train_unique_sizes = set()
test_unique_sizes = set()

train_class_distribution_images = dict()
with os.scandir(train_path) as entries:
    for i in entries:
        sub_dir = Path(i.path)
        image_count = 0
        if i.is_dir():
            print(f"In directory {i}")
            for item in sub_dir.iterdir():
                if item.is_file() and item.suffix.lower() == '.png':
                    image_count+=1
                    with Image.open(item) as img:
                        train_unique_sizes.add(img.size)
        train_class_distribution_images[i.name]=image_count
print(sum(train_class_distribution_images.values()))


test_image_count=0
if test_path.is_dir():
    for item in test_path.iterdir():
        if item.is_file() and item.suffix.lower() == '.png':
            test_image_count+=1
            with Image.open(item) as img:
                test_unique_sizes.add(img.size)
print(test_image_count)

print("Unique Train Sizes",len(train_unique_sizes))
print("Unique Test Sizes",len(test_unique_sizes))

# Class Distribution across train and test
train_class_distribution_images = {
    k: train_class_distribution_images[k] 
    for k in sorted(train_class_distribution_images.keys(), key=int)
}

test_class_distribution_images = dict()
for i in pd.unique(test_csv["ClassId"]):
    test_class_distribution_images[i]=len(test_csv[test_csv["ClassId"] == i])

test_class_distribution_images = {
    k: test_class_distribution_images[k] 
    for k in sorted(test_class_distribution_images.keys(), key=int)
}

# Plots for Class Distribution
plt.figure()
plt.bar(train_class_distribution_images.keys(),train_class_distribution_images.values())
plt.title("Class Distribution - Train Dataset")
plt.tight_layout()
plt.savefig("Outputs/Class_Distribution_Over_Train.png")
plt.close()

plt.figure()
plt.bar(test_class_distribution_images.keys(),test_class_distribution_images.values())
plt.title("Class Distribution - Test Dataset")
plt.tight_layout()
plt.savefig("Outputs/Class_Distribution_Over_Test.png")
plt.close()

# Plotting image resolution in a graph
# Train set graph

def get_imgres(unique_sizes):
    unique_sizes = list(unique_sizes)
    X = [0 for i in range(len(unique_sizes))]
    Y = [0 for i in range(len(unique_sizes))]
    for i in range(len(unique_sizes)):
        X[i] = unique_sizes[i][0]
        Y[i] = unique_sizes[i][1]
    return X,Y

train_X,train_Y = get_imgres(train_unique_sizes)
test_X,test_Y = get_imgres(test_unique_sizes)

def plot_imgres(X,Y,title):
    # Normal Graph
    plt.figure()
    plt.scatter(X,Y, color='purple', s=10, alpha=0.7, edgecolor='black')
    plt.title("Train Set - Image Resolution Graph")
    plt.tight_layout()
    plt.savefig("Outputs/Train_Set_ImgRes_Graph")
    plt.close()

    # Bounding Box Graph
    fig, ax = plt.subplots(figsize=(8, 8))
    max_dim = max(max(X), max(Y))
    canvas_size = max_dim * 1.1
    ax.set_xlim(0, canvas_size)
    ax.set_ylim(0, canvas_size)
    ax.set_aspect('equal')
    ax.axis('off')

    unique_sizes = list(set(zip(X, Y)))
    color_map = plt.colormaps.get_cmap('tab20') if hasattr(plt, 'colormaps') else plt.cm.get_cmap('tab20', len(unique_sizes))
    size_to_color = {size: color_map(i / max(1, len(unique_sizes) - 1)) for i, size in enumerate(unique_sizes)}

    for w, h in zip(X, Y):
        x_min = (canvas_size - w) / 2
        y_min = (canvas_size - h) / 2
        box_color = size_to_color[(w, h)]
        rect = Rectangle(
            (x_min, y_min), w, h,
            linewidth=1.0,
            edgecolor=box_color,
            facecolor='none',
            alpha=0.4
        )
        ax.add_patch(rect)

    outer_rect = Rectangle(
        (canvas_size * 0.05, canvas_size * 0.05), 
        canvas_size * 0.9, canvas_size * 0.9,
        linewidth=4, 
        edgecolor='#333333', 
        facecolor='none', 
        alpha=0.9
    )

    ax.add_patch(outer_rect)
    plt.tight_layout()
    plt.savefig("Outputs/"+title+".png", dpi=300, bbox_inches='tight')
    plt.close()

plot_imgres(train_X,train_Y,"Train_BoundingBox_IMGRES")
plot_imgres(test_X,test_Y,"Test_BoundingBox_IMGRES")