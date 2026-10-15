# Acne04-Seg

Dự án nghiên cứu phân đoạn tổn thương mụn trên ảnh da mặt bằng các mô hình YOLO segmentation, tập trung vào khả năng khái quát hóa giữa tập dữ liệu ACNE04 và các tập dữ liệu ngoài miền. Repository lưu script chuẩn hóa dữ liệu, các bảng kết quả benchmark và tài liệu nghiên cứu liên quan.

Script tiền xử lý tạo hai bộ dữ liệu benchmark ngoài miền:

- **Wrinkles/Acne**: giữ các polygon thuộc lớp mụn, loại polygon nếp nhăn.
- **Whitehead/Acne**: gộp các lớp tổn thương mụn về một lớp `acne`.

Ở cả hai bộ, ảnh không có nhãn mụn, ảnh không có file nhãn hợp lệ và ảnh trùng byte được loại bỏ. Nhãn đầu ra theo định dạng polygon segmentation của YOLO.

> **Phạm vi hiện tại:** `preprocess_datasets.py` và `evaluate_model.py` hiện có cùng nội dung; cả hai đều tiền xử lý dữ liệu, chưa chạy huấn luyện hay đánh giá mô hình. Các kết quả benchmark trong `benchmarks/` là artifact đã lưu sẵn. `preprocess_datasets.ipynb` hiện là notebook rỗng.

## Cấu trúc thư mục

```text
acne04-seg/
├── benchmarks/
│   ├── academic_failure_modes.csv
│   ├── academic_generalization_gap.csv
│   ├── academic_model_complexity.csv
│   ├── academic_table2_instance_metrics.csv
│   ├── academic_table3_image_level.csv
│   ├── academic_table4_wilcoxon_tests.csv
│   └── figure3_external_generalization_dice.png
├── dataset/                          # Dữ liệu đầu vào; cần tự chuẩn bị, không commit
│   ├── Acne Wrinkles Segmentation.v1i.yolov8/
│   └── acne-whitehead.v1i.yolov8/
├── clean_external_whitehead/         # Được tạo bởi script tiền xử lý
│   ├── images/
│   ├── labels/
│   └── clean_whitehead.yaml
├── clean_external_wrinkles/          # Được tạo bởi script tiền xử lý
│   ├── images/
│   ├── labels/
│   └── clean_wrinkles.yaml
├── evaluate_model.py
├── preprocess_datasets.py
├── preprocess_datasets.ipynb          # Hiện rỗng
├── README.md
├── Mô tả dataset.docx
├── MVIPIT2026___Cross_Dataset_Evaluation.pdf
└── NGHIÊN CỨU VÀ PHÁT TRIỂN HỆ THỐNG AI HỖ TRỢ BÁC SĨ CHẨN ĐOÁN
    BỆNH LÝ DA MẶT TỪ HÌNH ẢNH YOLO.pptx
```

Các thư mục dữ liệu, file nén và kết quả sinh ra bị loại khỏi Git theo `.gitignore`; chúng không nhất thiết có sẵn sau khi clone repository. Nội dung `benchmarks/` là các bảng CSV và hình tổng hợp kết quả đã lưu.

## Cài đặt môi trường

Yêu cầu Python **3.8 trở lên**. Repository hiện không có `requirements.txt`, `pyproject.toml` hoặc manifest phụ thuộc. Script tiền xử lý chỉ cần thư viện ngoài `PyYAML`; các thư viện còn lại được sử dụng thuộc Python Standard Library.

Trên Windows PowerShell, tạo môi trường ảo và cài phụ thuộc:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install PyYAML
```

Nếu PowerShell không tìm thấy lệnh `py`, cài Python từ python.org hoặc Microsoft Store, rồi mở terminal mới và chạy lại.

## Cách chạy (Usage)

### 1. Chuẩn bị dữ liệu

Đặt hai bộ dữ liệu thô bên dưới thư mục `dataset/` ở thư mục gốc project. Mỗi bộ cần có file YAML cấu hình Ultralytics và các ảnh cùng nhãn segmentation YOLO tương ứng. Script tìm đệ quy file `.yaml` có đường dẫn chứa `wrinkle` và `whitehead` (hoặc `acne and whitehead`).

Ví dụ cấu trúc:

```text
dataset/
├── Acne Wrinkles Segmentation.v1i.yolov8/
│   ├── data.yaml
│   ├── train/images/
│   ├── train/labels/
│   └── ...
└── acne-whitehead.v1i.yolov8/
    ├── data.yaml
    ├── train/images/
    ├── train/labels/
    └── ...
```

Lưu ý: mã nguồn đang dùng đường dẫn `dataset/` (số ít). Nếu dữ liệu được giải nén vào `datasets/` (số nhiều), hãy đổi tên hoặc đặt các thư mục bộ dữ liệu vào `dataset/` trước khi chạy. Tên thư mục/file YAML cần giữ từ khóa tương ứng để script nhận diện đúng.

### 2. Tiền xử lý benchmark ngoài miền

Chạy từ thư mục gốc project:

```powershell
python preprocess_datasets.py
```

Script tạo `clean_external_wrinkles/` và `clean_external_whitehead/`, mỗi thư mục gồm `images/`, `labels/` và file YAML cấu hình. Tên ảnh được đánh số lại; nhãn mụn được chuẩn hóa về class ID `0`.

> **Cảnh báo:** mỗi lần chạy, script xóa và tạo lại hai thư mục đầu ra này. Sao lưu dữ liệu tự thêm vào các thư mục đó trước khi chạy lại.

Các file YAML đầu ra hiện trỏ `train`, `val` và `test` cùng về `images/`. Đây là cấu hình benchmark/inference hiện tại, không phải cách chia tập độc lập để huấn luyện hoặc đánh giá không thiên lệch.

### 3. Xem kết quả benchmark đã lưu

Các kết quả tổng hợp có sẵn trong `benchmarks/`, ví dụ:

- `academic_table2_instance_metrics.csv`: Precision, Recall và mAP.
- `academic_table3_image_level.csv`: Dice và IoU theo ảnh.
- `academic_generalization_gap.csv`: khoảng cách tổng quát hóa giữa miền trong và ngoài.
- `academic_failure_modes.csv`: tỷ lệ bỏ sót hoàn toàn.
- `academic_model_complexity.csv`: độ phức tạp mô hình và checkpoint được ghi nhận.
- `academic_table4_wilcoxon_tests.csv`: kiểm định Wilcoxon và hiệu chỉnh Holm.

`evaluate_model.py` hiện là bản sao của `preprocess_datasets.py`; chạy file này cũng tiền xử lý dữ liệu và ghi đè các thư mục đầu ra, không tạo lại các bảng benchmark hoặc thực hiện inference. Hiện repository chưa cung cấp lệnh huấn luyện/đánh giá mô hình có thể chạy end-to-end.
