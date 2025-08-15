import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

class SpectrogramDataset(Dataset):
    """Custom Dataset for chunked spectrograms."""
    def __init__(self, csv_path, img_dir, mlb, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.mlb = mlb

        df = pd.read_csv(csv_path)
        
        self.samples = []
        all_files = os.listdir(img_dir)
        
        print(f"Processing {len(df)} rows from CSV")
        print(f"Found {len(all_files)} files in {img_dir}")
        
        for idx, row in df.iterrows():
            base_filename = row['filename']
            # Clean up labels by stripping whitespace
            labels = [label.strip() for label in row['labels'].split(',')]
            
            # Find all chunks for this base filename
            chunk_files = [f for f in all_files if f.startswith(base_filename.replace('.wav', '')) and f.endswith('.png')]
            
            if idx == 0:  # Debug first row
                print(f"First row - fname: {base_filename}")
                print(f"First row - labels: {labels}")
                print(f"First row - found chunks: {len(chunk_files)}")
            
            for chunk_file in chunk_files:
                img_path = os.path.join(img_dir, chunk_file)
                self.samples.append((img_path, labels))
        
        print(f"Total samples created: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, labels_str = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB') # Convert to 3 channels
        
        if self.transform:
            image = self.transform(image)
            
        # Debug label processing
        if idx == 0:  # Debug first sample
            print(f"Sample 0 - labels_str: {labels_str}")
            label_tensor = self.mlb.transform([labels_str])[0]
            print(f"Sample 0 - label_tensor shape: {label_tensor.shape}")
            print(f"Sample 0 - label_tensor sum: {label_tensor.sum()}")
            print(f"Sample 0 - positive indices: {np.where(label_tensor)[0]}")
        else:
            label_tensor = self.mlb.transform([labels_str])[0]
            
        label_tensor = torch.from_numpy(label_tensor.astype(np.float32))
                
        return image, label_tensor

def create_dataloaders(csv_path, img_dir, batch_size=32, val_split=0.2 , height=192 , width=668):
    """Creates training and validation DataLoaders."""
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create and fit the MultiLabelBinarizer
    df = pd.read_csv(csv_path)
    # Prepare labels for the binarizer: a list of lists of strings
    labels = [ [label.strip() for label in row.split(',')] for row in df['labels']]
    mlb = MultiLabelBinarizer()
    mlb.fit(labels)

    # Create dataset
    dataset = SpectrogramDataset(csv_path=csv_path, img_dir=img_dir, mlb=mlb, transform=transform)
    
    # Split dataset
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, mlb

if __name__ == '__main__':
    # Example usage:
    CSV_PATH = '/home/baraa/Desktop/testroch/DSED_strong_label/weak_labels.csv'
    IMG_DIR = '/home/baraa/Desktop/testroch/DSED_strong_label/mel/'
    
    # Check if files exist before proceeding
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
    elif not os.path.exists(IMG_DIR):
        print(f"Error: {IMG_DIR} not found.")
    else:
        train_loader, val_loader, mlb = create_dataloaders(CSV_PATH, IMG_DIR)
        
        print(f"Created {len(train_loader)} training batches and {len(val_loader)} validation batches.")
        print(f"Number of classes: {len(mlb.classes_)}")
        print(f"Classes: {mlb.classes_}")
        
        # Check a sample from the train_loader
        images, labels = next(iter(train_loader))
        print(f"Image batch shape: {images.shape}")
        print(f"Labels batch shape: {labels.shape}")
        print(f"Example labels tensor: {labels[0]}")
        # You can see the original labels for this tensor with:
        print(f"Original labels for first item: {mlb.inverse_transform(labels.numpy())[1]}")
