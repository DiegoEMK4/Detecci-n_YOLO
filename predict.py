from ultralytics import YOLO

# Cargar modelo
model = YOLO("plecostomus-seg/yolov8n-seg-custom/weights/best.pt")
img_path = "dataset/valid/images/1.jpg" 
results = model(img_path)
for r in results:
    r.show()
