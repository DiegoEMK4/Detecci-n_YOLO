from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

# Entrenar
model.train(
    data="dataset/dataset.yaml",
    epochs=100,
    imgsz=640,       
    batch=8,
    project="plecostomus-seg", 
    name="yolov8n-seg-custom" 
)
