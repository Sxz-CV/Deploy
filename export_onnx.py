from ultralytics import YOLO

PT = r"E:\desktop\deploy\weights\best.pt"
IMGSZ = 640
OPSET = 12

model = YOLO(PT)

onnx_path = model.export(format="onnx", imgsz=IMGSZ, opset=OPSET)
