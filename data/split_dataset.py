import os
import csv
import shutil
import random

def split_dataset(
    original_dir="synthetic_dataset",
    output_base="synthetic_dataset",
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    seed=42
):
    # Load all labels
    labels_path = os.path.join(original_dir, "labels.csv")
    with open(labels_path, "r") as f:
        reader = list(csv.DictReader(f))
        all_samples = [row for row in reader]

    # Shuffle for randomness
    random.seed(seed)
    random.shuffle(all_samples)

    # Compute split sizes
    total = len(all_samples)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)
    test_end = val_end + int(test_ratio * total)
    assert test_end == total, "Train, val, and test ratios must sum to 1."

    splits = {
        "train": all_samples[:train_end],
        "val": all_samples[train_end:val_end],
        "test": all_samples[val_end:]
    }

    for split_name, split_data in splits.items():
        # Create train, test, valid folders
        split_dir = os.path.join(output_base, split_name)
        images_dir = os.path.join(split_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        # Write labels
        with open(os.path.join(split_dir, "labels.csv"), "w", newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["frame", "x", "y"])
            writer.writeheader()

            for row in split_data:
                # Copy image file
                src_img = os.path.join(original_dir, "images", f"frame_{row['frame']}.png")
                dst_img = os.path.join(images_dir, f"frame_{row['frame']}.png")
                shutil.copyfile(src_img, dst_img)

                # Write label row
                writer.writerow(row)

    print("Dataset split into train/val/test successfully!")

if __name__ == "__main__":
    split_dataset()
