from ultralytics import YOLO
import json
import glob
import os

image_files = []
for root, dirs, files in os.walk("./datasets/val/"):
    for file in files:
        if file.endswith(".png"):
            image_files.append(os.path.join(root, file))
            
for root, dirs, files in os.walk("./datasets/train/"):
    for file in files:
        if file.endswith(".png"):
            image_files.append(os.path.join(root, file))

batch_size = 100


if not image_files:
    print("No images found in ./datasets/val/")
else:
    model = YOLO("best-v8.pt")
    results = model.predict(image_files, stream=True)
    output_data_list = []
    for i in range(0, len(image_files), batch_size):
        batch = image_files[i:i+batch_size]
        results = model.predict(batch, stream=True)
        for result in results:
            # process the results
            class_names = result.names
            probabilities = result.probs.data
            max_prob_index = result.probs.top1
            image_path = result.path.split("/")[-1]
            max_prob_class_name = class_names[max_prob_index]

            output_data = {
                "image_path": image_path,
                "class_names": class_names,
                "probabilities": probabilities.tolist(), 
                "max_prob_index": int(max_prob_index),
                "max_prob_class_name": max_prob_class_name
            }

            output_data_list.append(output_data)


    with open("output-v8-trainval.json", "w") as outfile:
        json.dump(output_data_list, outfile, indent=4)