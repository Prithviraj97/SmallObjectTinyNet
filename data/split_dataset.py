import os
import csv
import shutil
import random

class DatasetSplitter:
    def __init__(self, dataset_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        self.dataset_dir = dataset_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split_dataset(self):
        # Check if the dataset directory exists
        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError(f"Dataset directory {self.dataset_dir} does not exist.")

        # Create directories for train, val, and test sets
        train_dir = os.path.join(self.dataset_dir, "train")
        val_dir = os.path.join(self.dataset_dir, "val")
        test_dir = os.path.join(self.dataset_dir, "test")

        for dir in [train_dir, val_dir, test_dir]:
            os.makedirs(dir, exist_ok=True)

        # Get all image files
        image_files = [f for f in os.listdir(os.path.join(self.dataset_dir, "images")) if f.endswith('.png')]
        random.shuffle(image_files)

        # Calculate the number of files for each set
        total_files = len(image_files)
        train_count = int(total_files * self.train_ratio)
        val_count = int(total_files * self.val_ratio)
        
        # Split the dataset
        train_files = image_files[:train_count]
        val_files = image_files[train_count:train_count + val_count]
        test_files = image_files[train_count + val_count:]

        # Move files to respective directories
        for file in train_files:
            shutil.move(os.path.join(self.dataset_dir, "images", file), os.path.join(train_dir, file))
        
        for file in val_files:
            shutil.move(os.path.join(self.dataset_dir, "images", file), os.path.join(val_dir, file))
        
        for file in test_files:
            shutil.move(os.path.join(self.dataset_dir, "images", file), os.path.join(test_dir, file))

        print("Dataset split into train, validation, and test sets.")