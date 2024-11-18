import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
import torch
import torchvision
from datetime import datetime
import os
from ultralytics import YOLO  # For YOLO v11, but if you're not using Ultralytics, replace this with your own model definition.
from collections import defaultdict
import logging

base_image_dir = "..\..\yolov11_breaker\ds\img"
base_output_image_dir = "..\..\yolov11_breaker\ds\patched_images"
log_dir = "../logs"


curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"patcher_{curr_time}.log")

logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


def log_and_print(message):
    print(message)
    logging.info(message)

# Load an example image
def load_image(image_path):

    log_and_print(f"Loading image from {image_path}")

    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Overlay patch onto image without modifying the original image
def apply_patch_direct(image, patch, location):
    patched_image = image.clone()
    h, w = patch.shape[1:3]
    patched_image[:, :, location[0]:location[0]+h, location[1]:location[1]+w] = patch
    return patched_image

# Visualize the patched image
def visualize_image(image_tensor):
    image_np = image_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    plt.imshow(np.clip(image_np, 0, 1))
    plt.axis('off')
    plt.show()


def train_patch(image, target_class, patch_size=(50, 50), num_steps=500, lr=0.05):
    
    # Initialize random patch with the required size
    patch = torch.rand((3, *patch_size), requires_grad=True)
    optimizer = optim.Adam([patch], lr=lr)

    # Counter for consecutive steps where predicted label equals target class
    consecutive_count = 0
    max_consecutive = 5  # Number of consecutive steps to terminate on matching target class

    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Apply patch directly without altering the base image
        patched_image = apply_patch_direct(image, patch, (0, 0))  # Place at the top-left corner
        
        # Forward pass through the model
        output = model(patched_image)
        
        # Calculate loss to maximize the target class probability
        loss = -output[0][target_class]  # Negative because we want to maximize the target class probability
        loss.backward()
        
        # Update the patch
        optimizer.step()
        
        # Get the predicted label
        _, predicted_idx = torch.max(output, 1)
        predicted_label = labels[predicted_idx.item()]
        
        
        # Print training progress every 50 steps
        if step % 50 == 0:
            _, predicted_idx = torch.max(output, 1)
            predicted_label = labels[predicted_idx.item()]
            log_and_print(f"Step {step}, Loss: {loss.item()}, Predicted label: {predicted_label}")


        # Check if the target class is matched for 5 consecutive steps, this is checked more frequently        
        if step % 10 == 0:
            # If we have 5 consecutive steps where predicted label equals target class, stop training

            if predicted_idx.item() == target_class:
                    consecutive_count += 1
            else:
                consecutive_count = 0  # Reset the counter if it doesn't match

            if consecutive_count >= max_consecutive:
                log_and_print(f"Target class matched for {max_consecutive} consecutive steps. Stopping training.")
                break

    return patch, predicted_label


def image_patcher(image, target_class, patch_size):
    patch, predicted_label = train_patch(image, target_class, patch_size)
    patched_image = apply_patch_direct(image, patch, (0, 0))
    return patched_image, predicted_label


def persist_patched_image(image, target_class, patch_size, save_path):
    patched_image, predicted_label = image_patcher(image, target_class, patch_size)
    # visualize_image(patched_image)
    patched_image_np = patched_image.squeeze().permute(1, 2, 0).detach().cpu().numpy()

    save_path_with_adv_label = save_path.replace("<ADV_LABEL>", predicted_label)

    label_data = save_path_with_adv_label.split("/")[1].split("_")
    true_label = label_data[0]
    base_label = label_data[1]  
    predicted_label = label_data[2]

    # Log the true lable, predicted label and the adversarial label
    log_and_print(f"True label: {true_label}, Base label: {base_label}, Adversarial label: {predicted_label}")

    plt.imsave(save_path_with_adv_label, np.clip(patched_image_np, 0, 1))



def iterate_over_images(model, patch_size):

    start_time = datetime.now()
    log_and_print(f"Patching start time: {start_time}")

    labels = ['bicycle', 'bridge', 'bus', 'car', 'chimney', 'crosswalk', 'hydrant', 'motorcycle', 'mountain', 'other', 'palm', 'traffic', 'stairs']
    image_infos = [[os.path.join(base_image_dir, fname), fname] for fname in os.listdir(base_image_dir) if fname.endswith(('.png', '.jpg', '.jpeg'))]

    
    true_label_counts = defaultdict(lambda :0)
    max_records_per_class = 100

    for image_info in image_infos:
        filepath = image_info[0]
        filename = image_info[1]

        true_label = filename.split(" ")[0]
        if true_label_counts[true_label] >= max_records_per_class:
            log_and_print(f"Skipping {true_label} as it has reached the max records per class")
            continue

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image = load_image(filepath)
        output = model(image)
        _, predicted_idx = torch.max(output, 1)

        predicted_label = labels[predicted_idx.item()]

        adversarial_label = (labels.index(predicted_label) + 3) % len(labels)

        persist_patched_image(image, adversarial_label, patch_size, f"{base_output_image_dir}/{true_label}_{predicted_label}_<ADV_LABEL>_{timestamp}.png")

        true_label_counts[true_label] += 1

    
    end_time = datetime.now()
    log_and_print(f"Patching end time: {end_time}")

    total_time = end_time - start_time
    log_and_print(f"Total time taken: {total_time}")



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model architecture (For YOLOv11 or similar, you would load it like this)
    model = YOLO("..\models\YOLO_v11\yolo_v11.pt").to(device)  # Use the Ultralytics YOLO class

    # Load the saved state dict
    checkpoint = torch.load("..\models\YOLO_v11\yolo_v11.pt", map_location=device)

    # If the saved file is only the model weights (state_dict), you need to load them manually
    model = checkpoint['model'].to(device).to(torch.float32)  # Load the model to the device and set the datatype to float32

    # Now set the model in evaluation mode
    model.eval()

    # Define the class labels for the YOLO model
    labels = ['bicycle', 'bridge', 'bus', 'car', 'chimney', 'crosswalk', 'hydrant', 'motorcycle', 'mountain', 'other', 'palm', 'traffic', 'stairs']


    # Define transformation for input images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    patch_size=(50, 50)
    iterate_over_images(model, patch_size)
    