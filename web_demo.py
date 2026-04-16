import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1. 页面基本设置
st.set_page_config(page_title="纺织品瑕疵检测系统", page_icon="🧶")

# 2. 标题和简介
st.title("🧶 AI 纺织品表面瑕疵检测系统")
st.write("基于 YOLOv8 深度学习模型 | 作者：Hasee8pro")

# 3. 侧边栏：加载模型与参数配置
st.sidebar.header("🛠️ 参数配置")
conf_threshold = st.sidebar.slider("检测置信度 (Confidence)", 0.0, 1.0, 0.4, 0.05)


# 加载模型 (使用缓存装饰器，防止每次刷新都重新加载模型，速度更快)
@st.cache_resource
def load_model():
    # 替换成你最好的模型路径
    model_path = 'results/weights/best.pt'
    return YOLO(model_path)


try:
    model = load_model()
    st.sidebar.success("✅ 模型加载成功！")
except Exception as e:
    st.sidebar.error(f"模型加载失败: {e}")

# 4. 文件上传区
uploaded_file = st.file_uploader("📤 请上传一张布料图片", type=['jpg', 'png', 'jpeg'])

# 5. 开始检测逻辑
if uploaded_file is not None:
    # 将上传的文件转为图片格式
    image = Image.open(uploaded_file)

    # 布局：两列，左边原图，右边结果
    col1, col2 = st.columns(2)

    with col1:
        st.header("原始图片")
        st.image(image, use_container_width=True)

    # 这里的按钮是“点击开始检测”
    if st.button("🔍 开始智能检测"):
        with st.spinner('正在分析布料纹理...'):
            # YOLO 预测
            results = model.predict(source=image, conf=conf_threshold)

            # 画框
            res_plotted = results[0].plot()

            # 由于 OpenCV/YOLO 使用 BGR 格式，Streamlit 使用 RGB，需要转换一下颜色
            res_rgb = res_plotted[:, :, ::-1]

        with col2:
            st.header("检测结果")
            st.image(res_rgb, use_container_width=True)

        # 6. 显示具体的检测数据
        st.success("检测完成！发现以下目标：")

        # 统计检测到的瑕疵数量
        boxes = results[0].boxes
        if len(boxes) == 0:
            st.info("🎉 完美！未检测到任何瑕疵。")
        else:
            for box in boxes:
                # 获取类别名称
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                st.warning(f"⚠️ 发现瑕疵：**{cls_name}** (置信度: {conf:.2f})")

else:
    st.info("请在上方上传图片以开始。")