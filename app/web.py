import streamlit as st
import cv2
import numpy as np
import torch
import os
from ultralytics import YOLO
import pandas as pd
import altair as alt
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel

st.set_page_config(page_title="Chẩn đoán Ung thư Da", layout="wide", page_icon="🩺")

for key in ["processed_img", "processed_alt", "prediction_results", "steps"]:
    if key not in st.session_state:
        st.session_state[key] = None
# load models 
@st.cache_resource
def load_yolo_model():
    return YOLO("best.pt")

@st.cache_resource
def load_blip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "./models/blip" if os.path.exists("./models/blip") else "Salesforce/blip-image-captioning-base"
    
    p = BlipProcessor.from_pretrained(model_path, use_fast=True)
    m = BlipForConditionalGeneration.from_pretrained(model_path).to(device)
    m.eval()
    return p, m

@st.cache_resource
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "./models/clip" if os.path.exists("./models/clip") else "openai/clip-vit-base-patch32"
    
    m = CLIPModel.from_pretrained(model_path).to(device)
    p = CLIPProcessor.from_pretrained(model_path, use_fast=True)
    m.eval()
    return m, p

# preprocessing: CLAHE + Sharpen
def preprocess_pipeline(image):
    steps = {}

    # CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    clahe_img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    steps["CLAHE"] = clahe_img

    # Sharpen
    blur = cv2.GaussianBlur(image, (5,5), 0)
    sharpen = cv2.addWeighted(image, 1.5, blur, -0.5, 0)
    steps["Sharpen"] = sharpen

    return steps, clahe_img, sharpen

def resize_for_yolo(image, target_size=640):
    h, w = image.shape[:2]
    
    r = min(target_size / h, target_size / w)
    new_unpad = int(round(w * r)), int(round(h * r))
    
    dw, dh = target_size - new_unpad[0], target_size - new_unpad[1]
    dw /= 2 
    dh /= 2
    
    img_resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    return img_padded

def describe_image(image, processor, model):
    try:
        inputs = processor(image, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50)
        return processor.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        return f"Lỗi mô tả ảnh: {str(e)}"

def is_skin_image(image, processor, model):
    try:
        labels = [
            "a dermoscopic skin lesion",
            "a mole close-up",
            "a selfie",
            "a person sitting",
            "random object"
        ]
        inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits_per_image
        probs = logits.softmax(dim=1)[0]
        return (probs[0] + probs[1]) > (probs[2:].sum())
    except:
        return True

# UI
st.title("PHÂN LOẠI UNG THƯ BCC VÀ MEL")

with st.sidebar:
    st.header("Cài đặt AI")
    conf_thresh_base = st.slider(
    "Ngưỡng tin cậy chẩn đoán",
    min_value=0.40,  
    max_value=0.95, 
    value=0.65,      
    step=0.05,
    help="Mức khuyến nghị là 0.65 để tránh bỏ sót các dấu hiệu ung thư sớm."
)
    st.info("Lưu ý: Ngưỡng lọc tóc càng thấp thì càng nhiều chi tiết mảnh bị xóa.")

# load models
try:
    yolo_model = load_yolo_model()
    blip_proc, blip_mod = load_blip()
    clip_mod, clip_proc = load_clip()
except Exception as e:
    st.error("Lỗi khi tải mô hình. ")
    st.stop()

uploaded_file = st.file_uploader("Tải lên ảnh tổn thương da", type=["jpg","png","jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_rgb = cv2.cvtColor(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    img_rgb = resize_for_yolo(img_rgb)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Dữ liệu đầu vào")
        st.image(img_rgb, use_container_width=True)

        if st.button("Tiền xử lý"):
            try:
                steps, clahe_img, sharp_img = preprocess_pipeline(img_rgb)
                st.session_state.steps = steps
                st.session_state.processed_img = clahe_img
                st.session_state.processed_alt = sharp_img
                st.session_state.prediction_results = None
            except Exception as e:
                st.error("Lỗi tiền xử lý")
                st.exception(e)

    if st.session_state.steps is not None:
        with col2:
            st.subheader("CLAHE")
            st.image(st.session_state.processed_img, use_container_width=True)

        with col3:
            st.subheader("Sharpen")
            st.image(st.session_state.processed_alt, use_container_width=True)
        
        selected_img = img_rgb

        conf_thresh = conf_thresh_base

        if st.button("Thực hiện Dự đoán"):
            try:
                if not is_skin_image(selected_img, clip_proc, clip_mod):
                    st.error(" Đây không giống một bức ảnh y khoa về da!")
                    desc = describe_image(selected_img, blip_proc, blip_mod)
                    st.warning(f"AI nhận diện đây là: {desc}")
                else:
                    res = yolo_model.predict(selected_img)[0]
                    st.session_state.prediction_results = res
            except Exception as e:
                st.error("❌ Lỗi khi dự đoán")
                st.exception(e)

# results
if st.session_state.prediction_results is not None:
    st.divider()

    try:
        res = st.session_state.prediction_results

        if res.probs is None:
            st.error("❌ Không có dữ liệu xác suất (Mô hình đang là Object Detection thay vì Classification?)")
            st.stop()

        probs_np = res.probs.data.cpu().numpy()
        sorted_idx = np.argsort(probs_np)[::-1]

        conf = float(probs_np[sorted_idx[0]])
        label = res.names[int(sorted_idx[0])].upper()

        if conf < conf_thresh:
            st.subheader("Kết quả phân tích")
            st.warning(" **Phát hiện tổn thương da không xác định (Other/Unknown)**")
            
            st.markdown(f"""
            Hệ thống xác định đây là ảnh vùng da, nhưng các đặc điểm hình thái không đủ mạnh để xếp vào BCC hoặc MEL 
            (Độ tin cậy hiện tại: **{conf*100:.2f}%** < Ngưỡng cài đặt **{conf_thresh*100:.0f}%**).
            
            **Nguyên nhân có thể:**
            - Đây là nốt ruồi thông thường (NV), dày sừng lành tính (BKL), hoặc u sợi da.
            - Ảnh chụp chưa đủ rõ nét hoặc góc chụp bị lóa.
            """)
            
            with st.spinner("AI đang phân tích sâu hơn..."):
                extra_desc = describe_image(selected_img, blip_proc, blip_mod)
                st.info(f" **Mô tả hình thái AI:** {extra_desc}")
                
        else:
            st.subheader("Kết quả chẩn đoán chính xác")
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.metric("Chẩn đoán", label)
                st.metric("Độ tin cậy", f"{conf*100:.2f}%")
                if "MEL" in label:
                    st.error(" Cảnh báo: Melanoma (Nguy hiểm cao)")
                elif "BCC" in label:
                    st.warning(" Cảnh báo: BCC (Cần can thiệp y tế)")
                else:
                    st.success(" Đánh giá: Tổn thương da khác nhưng cần theo dõi)")
            with c2:
                chart_data = pd.DataFrame({
                    "Loại bệnh": [res.names[i] for i in range(len(res.names))],
                    "Xác suất": probs_np
                })
                
                chart = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X("Loại bệnh", sort='-y'),
                    y=alt.Y("Xác suất", scale=alt.Scale(domain=[0, 1])),
                    color=alt.condition(alt.datum['Xác suất'] > 0.5, alt.value('#ff4b4b'), alt.value('#31333f'))
                ).properties(height=250)
                
                st.altair_chart(chart, use_container_width=True)

        st.caption("Khuyến nghị: Kết quả chỉ mang tính chất tham khảo. Vui lòng gặp bác sĩ chuyên khoa để có chẩn đoán cuối cùng.")

    except Exception as e:
        st.error("Lỗi khi hiển thị kết quả")
        st.exception(e)