from ultralytics import YOLO

# Cargar modelo
model = YOLO("plecostomus-seg/yolov8n-seg-custom/weights/best.pt")
metrics = model.val()
print("Precisión mAP (cajas):", metrics.box.map)
print("Precisión mAP (máscaras):", metrics.seg.map)
