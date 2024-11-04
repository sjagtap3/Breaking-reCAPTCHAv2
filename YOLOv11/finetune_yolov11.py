from ultralytics import YOLO
import numpy as np
import pandas as pd

if __name__ == '__main__':
    model = YOLO("yolo11n-cls.pt")
    # results = model.train(data="datasets\captcha\\", epochs=100, imgsz=120, patience=10, batch=32, workers=20, save_period=10, device=0)

    epochs = [5, 10, 20]
    lr0 = [0.01, 0.001, 0.0001]  # initial learning rate
    dropout = [0.0, 0.1, 0.2]  # dropout regularization

    summary = pd.DataFrame(columns=["epochs", "lr0", "dropout", "top1acc"])

    i = 1
    for e in epochs:
        for l in lr0:
            for d in dropout:
                print("=======================================================================================")
                print(f"epochs: {e}, lr0: {l}, dropout: {d}")
                result = model.train(data="datasets\captcha\\", epochs=e, imgsz=120, batch=32, workers=20, device=0, lr0=l, dropout=d, name=f"grid_train_{i}")
                summary = pd.concat([summary, pd.DataFrame([{"epochs": e, "lr0": l,  "dropout": d, "top1acc": result.top1}])], ignore_index=True)
                i += 1
                print(summary)

    summary.to_csv("finetune_yolov11_results.csv", index=False)