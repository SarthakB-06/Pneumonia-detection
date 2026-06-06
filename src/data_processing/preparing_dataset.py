import os
import shutil
import random

def create_validation_set(base_dir='data/raw', val_split=0.2):
    """
    Moves a percentage of images from the training set to the validation set.

    Args:
        base_dir (str): The base directory where 'train' and 'val' folders are located.
        val_split (float): The percentage of training data to move to the validation set.
    """
    print("--- Starting Dataset Rebalancing ---")

    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')

    # Ensure the base validation directories exist
    os.makedirs(os.path.join(val_dir, 'NORMAL'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'PNEUMONIA'), exist_ok=True)

    for category in ['NORMAL', 'PNEUMONIA']:
        print(f"\nProcessing category: {category}")

        source_dir = os.path.join(train_dir, category)
        dest_dir = os.path.join(val_dir, category)

        # Get all image files, ignoring subdirectories
        images = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
        
        # --- This part is important to handle your existing small validation set ---
        # Move existing validation images back to train to ensure a clean, random split
        existing_val_images = [f for f in os.listdir(dest_dir) if os.path.isfile(os.path.join(dest_dir, f))]
        if existing_val_images:
            print(f"Moving {len(existing_val_images)} existing validation images back to the training set for a clean split.")
            for img in existing_val_images:
                shutil.move(os.path.join(dest_dir, img), os.path.join(source_dir, img))
            # Refresh the list of images in the source directory
            images = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

        num_images = len(images)
        num_to_move = int(num_images * val_split)

        print(f"Total images in train/{category}: {num_images}")
        print(f"Moving {num_to_move} images to val/{category}...")

        # Randomly select images to move
        files_to_move = random.sample(images, num_to_move)

        # Move the files
        for file_name in files_to_move:
            source_path = os.path.join(source_dir, file_name)
            dest_path = os.path.join(dest_dir, file_name)
            shutil.move(source_path, dest_path)

        print(f"Successfully moved {len(os.listdir(dest_dir))} images to val/{category}.")
        print(f"Remaining images in train/{category}: {len(os.listdir(source_dir))}")

    print("\n--- Dataset Rebalancing Complete ---")
    print("Please re-run the data exploration notebook to see the new distribution.")


if __name__ == '__main__':
    # The base directory is relative to the project root
    create_validation_set(base_dir='data/raw', val_split=0.2)
