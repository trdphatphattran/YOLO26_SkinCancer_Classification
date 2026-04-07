
# Skin Cancer Classification using YOLO26  

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![YOLO26](https://img.shields.io/badge/Model-YOLO26-green.svg)](https://docs.ultralytics.com/models/yolo26/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io)

## 📌 Tổng quan dự án  
Đề tài nghiên cứu và triển khai hệ thống **Nhận diện và Phân loại Ung thư biểu mô tế bào đáy (BCC) và Ung thư hắc tố (Melanoma)** dựa trên mô hình **YOLO26**. Dự án giải quyết bài toán sàng lọc sớm bệnh lý da liễu bằng Trí tuệ nhân tạo, hỗ trợ chẩn đoán nhanh với độ chính xác cao.

## 🚀 Tính năng nổi bật  
* **High Performance:** Huấn luyện trên bộ dữ liệu quy mô lớn (~36,000 ảnh).
* **State-of-the-art:** Sử dụng kiến trúc YOLO26X tối ưu hóa cho bài toán phân loại y tế.
* **Real-time Demo:** Giao diện Web trực quan xây dựng bằng **Streamlit**, cho phép chẩn đoán ngay lập tức từ ảnh tải lên.
* **Precision Medical:** Trả về kết quả phân loại kèm theo độ chính xác.  

## 📊 Kết quả huấn luyện  
Mô hình được huấn luyện trên GPU **NVIDIA L4** với cấu hình tối ưu:
* **Dataset Size:** 36,087 images (sau khi Augmentation).
* **Input Size:** 800x800 pixels.
* **Accuracy (Top-1):** **~93%**
* **Epochs:** 60.

### Result & Confusion Matrix  
**Result**  
<img width="695" height="694" alt="image" src="https://github.com/user-attachments/assets/78230c77-998f-4acb-8e6c-4d9f4b7b595a" />  

**Confusion Matrix**  
<img width="757" height="661" alt="image" src="https://github.com/user-attachments/assets/b29a74d9-1dea-47f1-b777-952c3011e1c4" />


## Công nghệ sử dụng  
* **Core:** Python, PyTorch, Ultralytics (YOLO26).  
* **Data Processing:** Roboflow, OpenCV, Pandas.  
* **Deployment:** Streamlit.  
* **Training:** Google Colab (L4 GPU).

## Demo Web  
### 1. Upload ảnh  
- Upload ảnh MEL, BCC hay others (ung thư khác).  
<img width="1421" height="736" alt="image" src="https://github.com/user-attachments/assets/9e0def64-554e-4b5a-9e06-4d902f33c1f1" />

### 2. Tiền xử lý ảnh  
- Ảnh được xử lý qua CLAHE và Sharpen.
<img width="1409" height="771" alt="image" src="https://github.com/user-attachments/assets/2539eb70-9360-4eb3-b4e0-414ddb0abdb2" />

### 3. Kết quả  
<img width="1419" height="245" alt="image" src="https://github.com/user-attachments/assets/0902c51b-83cb-43a6-8703-fccf954f3a93" />  



## 📂 Cấu trúc thư mục  
```text
├── web.py    # Code giao diện Streamlit
├── best.pt            # Trọng số mô hình tốt nhất sau khi train
├── train_CV_yolo26.ipynb # File Colab dùng để huấn luyện
└── README.md
```
## Hướng dẫn sử dụng  
### 1. Clone repository  
```python
git clone https://github.com/trdphatphattran/YOLO26_SkinCancer_Classification.git
```

### 2. Cài thư viện  
```python
pip install -r requirements.txt
```

### 3. Chạy streamlit  
```python
streamlit run app/web.py
```

## Thông tin liên hệ  

Nếu bạn có bất kỳ câu hỏi nào về dự án hoặc muốn hợp tác, vui lòng liên hệ với mình qua:

## 📬 Thông tin liên hệ

* **Họ và tên:** Trần Đại Phát
* **LinkedIn:** [Phat Tran](https://www.linkedin.com/in/phat-tran-9ba42a341/)
* **GitHub:** [trdphatphattran](https://github.com/trdphatphattran)
* **Email:** phattrandai15062005@gmail.com
* **Phone:** 0908647977 
