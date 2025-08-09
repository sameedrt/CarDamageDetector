import torch
from PIL import Image
import pathlib
import io
import os

pathlib.PosixPath = pathlib.WindowsPath
path = 'C:/Users/alisa/Desktop/Car_Damage_Detection/yolov5/best.pt'

custom_cache = r"C:/Users/alisa/torch_hub_cache"   # pick any writable location
os.makedirs(custom_cache, exist_ok=True)
torch.hub.set_dir(custom_cache)


model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path = path,
    force_reload=True,
    verbose=False
)

CLASS_NAMES = {
    0: "dent",
    1: "scratch",
    2: "crack",
    3: "glass shatter",
    4: "lamp broken",
    5: "tire flat"
}

def damage_detection(image_bytes: bytes) -> list[str]:
    #I need a list with all unique classes that are identified in the detection
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB") 
    results = model(img, size=640)
    detection = results.xyxy[0]


    labels = {CLASS_NAMES.get(int(cls), "unknown") for *_, cls in detection}
    return sorted(labels)