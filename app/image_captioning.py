import os
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from pillow import Image

#Load and clean captions, mapping to available images.
def load_captions(images_path, captions_file):
    captions = pd.read_csv(captions_file, sep=',', names=['image', 'caption'], engine='python')
    captions['image'] = captions['image'].str.strip().str.lower()

    available_files = {f.lower(): f for f in os.listdir(images_path)}

    def map_file(img):
        return available_files.get(img, None)

    captions['image'] = captions['image'].apply(map_file)
    captions = captions.dropna(subset=['image'])
    return captions

#Take one caption per image and return a random subset.
def prepare_subset(captions, sample_size=500):
    subset = (
        captions.groupby('image').head(1)
        .reset_index(drop=True)
        .sample(sample_size, random_state=42)
        .reset_index(drop=True)
    )
    return subset

#Return preprocessing transforms for image models.
def get_image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

#Custom PyTorch Dataset for image-caption pairs.
class CaptionDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image'])
        image = Image.open(img_path).convert('RGB')
        caption = row['caption']

        if self.transform:
            image = self.transform(image)

        return image, caption
