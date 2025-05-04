import torch
import cv2
import numpy as np
from ultralytics import YOLO

def load_depth_model():
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.eval()
    return model

def detect_objects(image, model):
    results = model(image)
    detected_objects = []
    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    labels = result.boxes.cls.cpu().numpy().astype(int)
    names = model.names

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        class_name = names[labels[i]]
        detected_objects.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "class_name": class_name
        })

    return detected_objects

def estimate_depth(image, detected_objects, depth_model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    input_tensor = torch.from_numpy(image).float().unsqueeze(0).permute(0, 3, 1, 2) / 255.0
    input_tensor = input_tensor.cuda() if torch.cuda.is_available() else input_tensor

    with torch.no_grad():
        depth_map = depth_model(input_tensor)

    depth_map = depth_map.squeeze().cpu().numpy()
    depth_map = cv2.resize(depth_map, (w, h))

    object_depths = []
    for obj in detected_objects:
        x1, y1, x2, y2 = obj["bbox"]
        obj_depth = np.mean(depth_map[y1:y2, x1:x2])

        object_depths.append({
            "class_name": obj["class_name"],
            "depth": obj_depth
        })

    object_depths.sort(key=lambda x: x['depth'])
    return object_depths[0]
