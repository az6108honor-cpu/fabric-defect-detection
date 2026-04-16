import os
from ultralytics import YOLO

if __name__ == '__main__':
    # ================= 🚀 GPU训练配置区 =================
    # 1. 获取当前项目根目录 (防止路径报错)
    ROOT_DIR = os.getcwd()

    # 2. 数据集配置路径
    # 指向 Textile... 文件夹里的 data.yaml
    DATA_YAML = r"C:\Users\hasee8pro\PycharmProjects\PythonProjectFabic\Textile defect datasts.v1i.yolov8\data.yaml"

    # 3. 模型底座
    # 继续使用你之前的 best_woven_highres.pt (如果文件不在 models 里，请修改这里)
    # 如果找不到这个文件，自动下载 yolov8s.pt（比 n 更强，GPU跑得动）
    MODEL_PATH = r'C:\Users\hasee8pro\PycharmProjects\PythonProjectFabic\models\best_woven_highres.pt'
    # ====================================================

    print("🚀 GPU模式启动：正在加载模型...")
    print(f"🕒 预计训练时长：2-4 小时 (GPU模式，RTX 4070)")
    print("💾 策略：每 1 轮自动存档，绝不白跑。")

    # 加载模型
    try:
        model = YOLO(MODEL_PATH)
    except:
        print(f"⚠️ 找不到 {MODEL_PATH}，自动切换为官方 yolov8s.pt 底座...")
        model = YOLO('yolov8s.pt')  # pretrained=True 已内含于 .pt 权重文件

    model.train(
        data=DATA_YAML,
        project='runs/detect',
        name='train_gpu_v1',

        # === 🚀 GPU核心参数 ===
        epochs=100,  # 100轮，配合 patience 早停自动收敛
        batch=16,    # RTX 4070 8GB显存，16跑得很稳
        workers=0,   # 【关键】Windows下必须为0，防止多进程报错

        # === 💾 存档机制 ===
        save=True,
        save_period=1,  # 每跑 1 轮就存一次！断电也不怕

        # === ⚙️ 训练策略 ===
        imgsz=640,   # GPU标准尺寸，速度与精度均衡
        device='0',  # 使用 GPU 0（RTX 4070 Laptop）
        patience=20, # 早停：连续20轮无提升自动停止，防过拟合
        close_mosaic=5,  # 最后 5 轮关闭增强，精修细节
        optimizer='auto',
        amp=False    # 关闭混合精度，稳定性优先
    )

    print("✅ 训练结束！请去 runs/detect/train_gpu_v1 查看结果。")
