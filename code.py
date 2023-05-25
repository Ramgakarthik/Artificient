import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms as T

torch.manual_seed(8)
np.random.seed(8)

img = cv2.imread(r"C:\Users\91812\Dropbox (Old)\My PC (LAPTOP-SMD1J3K9)\Documents\sample.png")


img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
transform = T.Compose([
    T.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    T.RandomRotation(degrees=5),
    T.RandomHorizontalFlip(p=0.5),
    T.CenterCrop((320, 640)),
    T.Resize((160, 320)),
    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.2)
])


augmented_img = transform(img)
augmented_img.save('sample_augmented.png')
