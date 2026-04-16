from ultralytics import YOLO
import os

# 1. 加载你刚练好的模型 (best.pt 是最好的那个)
model_path = 'results/weights/best.pt'
model = YOLO(model_path)

# 2. 随便找张图来测试
# (这里换成你随便一张布料图片的路径，或者验证集里的 valid/images/xxx.jpg)
image_path = r"C:\Users\hasee8pro\PycharmProjects\PythonProjectFabic\Textile defect datasts.v1i.yolov8\test\images\147_jpg.rf.74ff6990290c951615d41b1d41238590.jpg"
# 注意：这只是个目录，它会把里面第一张拿出来测，或者你指定具体某张 xxx.jpg

# 3. 开始预测，并把结果保存下来
# save=True 会把画好框的图存到 runs/detect/predict 里面
results = model.predict(source=image_path, save=True, max_det=1)

print("🎉 预测完成！快去 runs/detect/predict 文件夹看看效果！")