from ultralytics import YOLO
from glob import glob
from sklearn.metrics import precision_score, recall_score, accuracy_score, average_precision_score
import numpy as np

if __name__ == '__main__':
    # model = YOLO("./runs/classify/BEST_grid_train_21/weights/best.pt")
    model = YOLO("baseline_v8.pt")

    # results = model.predict("./datasets/inference.jpg", device=0, save=True, save_txt=True, save_conf=True, show_boxes=True, save_crop=True)
    # print(results)

    gts = []
    preds = []

    # get per class precision and recall
    path = "./datasets/captcha/val/*"
    for folder in glob(path):
        class_name = folder.split('/')[-1].split('\\')[-1]
        num_samples = len(glob(folder + "/*"))

        results = model.predict(folder, device=0, verbose=False)

        pred = [result.names[result.probs.top1] for result in results]
        pred_final = [p if p == class_name else "WRONG" for p in pred]

        gt = [class_name] * num_samples
        
        print(f"Class {class_name} precision: {precision_score(gt, pred_final, pos_label=class_name)}")
        print(f"Class {class_name} recall: {recall_score(gt, pred_final, pos_label=class_name)}")
        print(f"Class {class_name} accuracy: {accuracy_score(gt, pred_final)}")

        gts.extend(gt)
        preds.extend(pred_final)

    print(f"Macro Averaged precision: {precision_score(gts, preds, average='macro')}")
    print(f"Macro Averaged recall: {recall_score(gts, preds, average='macro')}")
    print(f"Overall accuracy: {accuracy_score(gts, preds)}")