from ultralytics import YOLO
import numpy as np
import pandas as pd
import torch

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)

    # model = YOLO("yolo_v11.pt").to(device)
    # model = YOLO('yolov8n-seg.pt').to(device)
    # model = YOLO("yolo11n.pt").to(device)
    model = YOLO("yolo11n-cls.pt").to(device)
    

    # print(model)

    # save_conf=True, show_boxes=True,\ save_txt=True, \ save=True, \

    results = model.predict("./assets/hydrant_hd.jpg",\
                                save_conf=True, \
                                show_boxes=True, \
                                save_txt=True, \
                                # save=True, \
                                save_crop=True, device='cpu')

    # results = model("https://ultralytics.com/images/bus.jpg") 

    print(results)