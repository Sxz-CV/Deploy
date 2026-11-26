from ultralytics import YOLO

print("加载模型...")
model = YOLO(r'E:\desktop\deploy\weights\best.pt')

print("开始推理...")
results = model.predict(
    source=r'E:\desktop\deploy\dataset\images\test',
    conf=0.5,
    save=True,
    project=r'E:\desktop\deploy\result',
    name='pred',
)

print(f"共处理 {len(results)} 帧")
print("完成")