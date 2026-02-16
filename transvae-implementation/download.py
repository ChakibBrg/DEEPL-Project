from datasets import load_dataset
import os

def main():
    output_dir = "./imagenet_1k_256_data"

    for split in ['train', 'val', 'test']:
        print(f"Processing {split}...")
        dataset = load_dataset("evanarlian/imagenet_1k_resized_256", split=split, streaming=True)
        
        for i, sample in enumerate(dataset):
            image = sample['image']
            label = sample['label']
            
            class_dir = os.path.join(output_dir, split, str(label))
            os.makedirs(class_dir, exist_ok=True)
            
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            image.save(os.path.join(class_dir, f"{i}.jpg"))
            
            if i % 1000 == 0:
                print(f"Saved {i} images from {split}")


if __name__ == '__main__':
    main()