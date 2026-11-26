from ultralytics import YOLO

def main():
    # 使用标准名称，去掉注释中的歧义
    model = YOLO('yolo26s-seg.pt') 
    
    model.train(
        data=r'E:\deepseek\Smoke\data.yaml',
        epochs=100,
        imgsz=640,
        batch=8,
        patience=35,
        # optimizer 保持默认 auto 或指定 MuSGD
        # lr0 建议降低到 0.001 或更低，或直接删除让框架用默认值
        lr0=0.001,  # 从 0.005 降低
        mosaic=0.5,
        copy_paste=0.3,
        project=r'E:\deepseek\Smoke\result', 
        name='save',
        workers=0,
        amp=True
    )

if __name__ == '__main__':
    main()