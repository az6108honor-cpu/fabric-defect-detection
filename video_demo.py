from ultralytics import YOLO
import cv2

# 1. 加载你的冠军模型
# 确保路径指向你刚才训练出来的 best.pt (在 runs/detect/train_safe_v1/weights/best.pt)
model_path = 'results/weights/best.pt'
model = YOLO(model_path)

# 2. 指定你的视频文件
video_path = "test_video.mp4"  # 确保文件名对得上
cap = cv2.VideoCapture(video_path)

# 获取视频的宽高，用来调整窗口大小
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"🎬 视频加载成功 ({width}x{height})！按 'q' 键退出")

while True:
    ret, frame = cap.read()
    if not ret:
        print("播放结束，循环播放中...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 循环播放
        continue

    # 3. 预测 (conf=0.3 稍微调低一点，看看能不能捕捉到褶皱或接缝)
    results = model.predict(source=frame, save=False, conf=0.3, show=False)

    # 4. 画框
    annotated_frame = results[0].plot()

    # 5. 缩小一点显示 (防止视频太大撑爆屏幕)
    # 如果视频是竖屏的，可以把宽高对调或者缩小比例
    display_frame = cv2.resize(annotated_frame, (int(width/2), int(height/2)))

    cv2.imshow("Detection Result", display_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'): # waitKey(30) 控制播放速度
        break

cap.release()
cv2.destroyAllWindows()