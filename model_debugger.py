from ultralytics import YOLO
import torch
from torchvision import transforms
from PIL import Image
import shap
from shap import KernelExplainer, force_plot
import numpy as np
import cv2
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the path to the best.pt file
model_path = "models/YOLO_Classification/train4/weights/yolo11n-cls.pt"
# os.path.join(script_dir, 'train4', 'weights', 'best.pt')

model = YOLO(model_path)

# print("%%", model)

results = model.predict("data/bus_tile_14.jpg", verbose=True)
# print(results)
print(results[0])
print(results[0].names)
print(results[0].probs.data)

    






