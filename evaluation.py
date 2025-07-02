import torch
import clip
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader # Import DataLoader for batch processing
from torchvision import transforms
import os
import tqdm # For progress bar
import time # Import time module for performance measurement
import random # For picking a random image at the end

# --- Configuration ---
# Path to your ImageNet-256 dataset
IMAGENET_DATASET_PATH = "/path/to/imagenet-256/versions/1"
BATCH_SIZE = 32 # Define batch size for efficient processing (adjust based on GPU memory)

# Load the model
# agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the CLIP model 
# The `preprocess` function handles resizing, cropping, normalization, etc.
model, preprocess = clip.load('ViT-L/14', device) # ViT-B/32  , ViT-L/14 , RN50
# Set the model to evaluation mode
model.eval()
print("CLIP model loaded.")

# --- Prepare the Dataset ---
# ImageNet-256 is expected to be a directory with subdirectories for each class.
# ImageFolder automatically handles this structure.
# The `preprocess` function from CLIP already handles all necessary transformations
# (resizing, cropping, normalization) and converts to a tensor.
try:
    # Load the ImageNet-256 dataset using ImageFolder
    imagenet_dataset = ImageFolder(root=IMAGENET_DATASET_PATH, transform=preprocess)
    print(f"ImageNet-256 dataset loaded from {IMAGENET_DATASET_PATH}.")
    print(f"Found {len(imagenet_dataset)} images across {len(imagenet_dataset.classes)} classes.")
except Exception as e:
    print(f"Error loading ImageNet-256 dataset: {e}")
    print("Please ensure the path is correct and the dataset is structured as: "
          "ROOT_DIR/class_name/image.jpg")
    exit() # Exit if dataset loading fails

# Create a DataLoader to iterate through the dataset in batches
# Set num_workers to 0 if you encounter issues with multiprocessing (common on some systems)
# Otherwise, os.cpu_count() is a good default.
data_loader = DataLoader(imagenet_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count())
print(f"DataLoader created with batch size: {BATCH_SIZE} and {os.cpu_count()} workers.")

# Prepare text inputs for all classes (only needs to be done once)
# Create prompts like "a photo of a [class_name]"
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in imagenet_dataset.classes]).to(device)
print(f"Prepared {len(imagenet_dataset.classes)} text prompts.")

# Calculate text features (only needs to be done once)
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True) # Normalize text features
print("Text features calculated and normalized.")

# --- Calculate Accuracy and Inference Time ---
correct_predictions = 0
total_images = 0
total_inference_time = 0.0 # Initialize total inference time

print("\nCalculating accuracy and inference time over the entire dataset...")
# Iterate through the dataset using the DataLoader
for images, true_labels in tqdm.tqdm(data_loader, desc="Processing batches"):
    # Move images to the correct device
    images = images.to(device)

    # Start timing for the current batch
    start_time = time.time()

    with torch.no_grad():
        # Encode the images to get their features
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True) # Normalize image features

        # Calculate similarity scores
        # (100.0 * ...) scales the logits for better softmax distribution
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # Get the top-1 predicted class index for each image in the batch
        # argmax returns the index of the maximum value
        predicted_indices = torch.argmax(similarity, dim=1)

    # End timing for the current batch
    end_time = time.time()
    total_inference_time += (end_time - start_time)

    # Compare predictions with true labels
    correct_predictions += (predicted_indices.cpu() == true_labels).sum().item()
    total_images += images.shape[0]

# Calculate the final accuracy
accuracy = (correct_predictions / total_images) * 100

# Calculate average inference time per sample
avg_inference_time_per_sample = (total_inference_time / total_images) * 1000 # Convert to milliseconds

# --- Print the Result ---
print(f"\n--- Results ---")
print(f"Total images processed: {total_images}")
print(f"Correct predictions: {correct_predictions}")
print(f"Accuracy on the entire dataset: {accuracy:.2f}%")
print(f"Total inference time: {total_inference_time:.4f} seconds")
print(f"Average inference time per sample: {avg_inference_time_per_sample:.4f} ms")


# --- Optional: Example Top 5 Predictions for a Random Image ---
# This part demonstrates the top 5 predictions for a single image,
# separate from the accuracy calculation.
if total_images > 0:
    random_idx = random.randint(0, len(imagenet_dataset) - 1)
    # The dataset already applies `preprocess` when an item is retrieved.
    image, class_id = imagenet_dataset[random_idx]
    true_class_name = imagenet_dataset.classes[class_id]

    # Add a batch dimension and move to device for single image inference
    image_input_single = image.unsqueeze(0).to(device)

    with torch.no_grad():
        image_features_single = model.encode_image(image_input_single)
        image_features_single /= image_features_single.norm(dim=-1, keepdim=True)
        similarity_single = (100.0 * image_features_single @ text_features.T).softmax(dim=-1)
        values_single, indices_single = similarity_single[0].topk(5)

    print(f"\n--- Example Top 5 Predictions for a random image (True class: '{true_class_name}') ---")
    for value, index in zip(values_single, indices_single):
        predicted_class_name = imagenet_dataset.classes[index]
        print(f"{predicted_class_name:>16s}: {100 * value.item():.2f}%")

