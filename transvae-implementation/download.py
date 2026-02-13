from datasets import load_dataset
import os

def main():
    # Define where you want to save the files
    output_dir = "./imagenet_1k_256_data"

    # Loop through splits and save images
    # We skip 'test' here since it has no labels, but you can include it if needed
    for split in ['train', 'val', 'test']:
        print(f"Processing {split}...")
        dataset = load_dataset("evanarlian/imagenet_1k_resized_256", split=split, streaming=True)
        
        for i, sample in enumerate(dataset):
            image = sample['image']
            label = sample['label']
            
            # Create a folder for the class label (e.g., ./imagenet_data/train/0/)
            class_dir = os.path.join(output_dir, split, str(label))
            os.makedirs(class_dir, exist_ok=True)
            
            # Save image (using index as filename)
            # Convert non-RGB images to RGB to avoid save errors
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            image.save(os.path.join(class_dir, f"{i}.jpg"))
            
            if i % 1000 == 0:
                print(f"Saved {i} images from {split}")


if __name__ == '__main__':
    main()