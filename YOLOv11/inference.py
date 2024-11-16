from ultralytics import YOLO
from glob import glob
from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np

if __name__ == '__main__':
    """CLassification model"""
    # model = YOLO("./runs/classify/BEST_grid_train_21/weights/best.pt")
    model = YOLO("models/baseline_v8.pt")

    # results = model.predict("./datasets/inference.jpg", device=0, save=True, save_txt=True, save_conf=True, show_boxes=True, save_crop=True)
    # print(results)

    gts = []
    preds = []
    prev_class = None

    # Define the folder path
    path = "./datasets/captcha/test/*"

    # Dictionary to map class names
    class_name_map = {
        "Cross": "Crosswalk",
        "Tlight": "Traffic Light",
        "Stair": "Stairs"
    }

    # Get all images
    images = glob(path)

    # Get unique classes
    unique_classes = set(img.split('/')[-1].split('\\')[-1].split('_')[0].split(" ")[0] for img in images)

    for class_name in unique_classes:
        # Map the names that are different
        mapped_class_name = class_name_map.get(class_name, class_name)

        # Get all images for the current class
        class_images = [img for img in images if class_name in img]
        
        # Get ground truth for the current class
        gt = [mapped_class_name] * len(class_images)

        # Get predictions for the current class
        results = model.predict(class_images, device=0, verbose=False, stream=True)
        pred = [result.names[result.probs.top1] for result in results]
        pred_final = [p if p == mapped_class_name else "WRONG" for p in pred]

        # Calculate precision, recall, and accuracy
        precision = precision_score(gt, pred_final, pos_label=mapped_class_name)
        recall = recall_score(gt, pred_final, pos_label=mapped_class_name)
        accuracy = accuracy_score(gt, pred_final)

        # Print the results
        print(f"Class {mapped_class_name} precision: {precision}")
        print(f"Class {mapped_class_name} recall: {recall}")
        print(f"Class {mapped_class_name} accuracy: {accuracy}")

        gts.extend(gt)
        preds.extend(pred_final)

    print(f"Macro Averaged precision: {precision_score(gts, preds, average='macro')}")
    print(f"Macro Averaged recall: {recall_score(gts, preds, average='macro')}")
    print(f"Overall accuracy: {accuracy_score(gts, preds)}")

    # # get per class precision and recall
    # path = "./datasets/captcha/val/*"
    # for folder in glob(path):
    #     class_name = folder.split('/')[-1].split('\\')[-1]
    #     num_samples = len(glob(folder + "/*"))

    #     results = model.predict(folder, device=0, verbose=False)

    #     pred = [result.names[result.probs.top1] for result in results]
    #     pred_final = [p if p == class_name else "WRONG" for p in pred]

    #     gt = [class_name] * num_samples
        
    #     print(f"Class {class_name} precision: {precision_score(gt, pred_final, pos_label=class_name)}")
    #     print(f"Class {class_name} recall: {recall_score(gt, pred_final, pos_label=class_name)}")
    #     print(f"Class {class_name} accuracy: {accuracy_score(gt, pred_final)}")

    #     gts.extend(gt)
    #     preds.extend(pred_final)

    # print(f"Macro Averaged precision: {precision_score(gts, preds, average='macro')}")
    # print(f"Macro Averaged recall: {recall_score(gts, preds, average='macro')}")
    # print(f"Overall accuracy: {accuracy_score(gts, preds)}")




    # """Segmentation model"""
    # # model = YOLO("runs/segment/train2/weights/best.pt")
    # model = YOLO("models/baseline_v8_seg.pt")
    
    # # need to change val as test in the yaml file so that ultralytics picks it up instead of the val set which is already used
    # metrics = model.val(data="segment.yaml", imgsz=120, batch=32, name=f"val")

    # print("Val set mAP50-95", metrics.seg.map)
    # print(metrics.box.maps)