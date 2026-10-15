# %%
import os
import sys
import shutil
import hashlib
import yaml
from pathlib import Path

print(f"Python executable: {sys.executable}")
print("Môi trường đã sẵn sàng để tiền xử lý dữ liệu ngoại suy.")

# %%
dataset_root = Path("dataset")

# 1. Quét tìm file data.yaml của bộ Wrinkles / Acne
wrinkles_yaml_path = None
whitehead_yaml_path = None

for y in dataset_root.rglob("*.yaml"):
    path_str = str(y).lower()
    if "wrinkle" in path_str and wrinkles_yaml_path is None:
        wrinkles_yaml_path = y
    elif ("whitehead" in path_str or "acne and whitehead" in path_str) and whitehead_yaml_path is None:
        whitehead_yaml_path = y

print("--- KẾT QUẢ TÌM KIẾM CẤU HÌNH DỮ LIỆU ---")
print(f"1. Wrinkles YAML:  {wrinkles_yaml_path}")
print(f"2. Whitehead YAML: {whitehead_yaml_path}")

# Đọc danh sách nhãn bộ Wrinkles
with open(wrinkles_yaml_path, "r", encoding="utf-8") as f:
    w_data = yaml.safe_load(f)
w_names = w_data.get("names")
print(f"\nNhãn gốc bộ Wrinkles: {w_names}")

# Xác định index của acne trong bộ Wrinkles
wrinkle_acne_ids = []
if isinstance(w_names, list):
    wrinkle_acne_ids = [i for i, name in enumerate(w_names) if "acne" in str(name).lower()]
elif isinstance(w_names, dict):
    wrinkle_acne_ids = [k for k, name in w_names.items() if "acne" in str(name).lower()]

print(f"=> Class ID được xác định là MỤN (bỏ qua wrinkles): {wrinkle_acne_ids}")

# Đọc danh sách nhãn bộ Whitehead
with open(whitehead_yaml_path, "r", encoding="utf-8") as f:
    wh_data = yaml.safe_load(f)
wh_names = wh_data.get("names")
print(f"\nNhãn gốc bộ Whitehead: {wh_names}")

# Whitehead/Acne gom tất cả các loại tổn thương mụn thành mụn
if isinstance(wh_names, list):
    whitehead_acne_ids = list(range(len(wh_names)))
elif isinstance(wh_names, dict):
    whitehead_acne_ids = list(wh_names.keys())
else:
    whitehead_acne_ids = [0, 1]

print(f"=> Class IDs gom thành tổn thương mụn trong bộ Whitehead: {whitehead_acne_ids}")

# %%
def get_file_md5(filepath: Path) -> str:
    """Tạo mã MD5 hash để lọc bỏ ảnh trùng lặp nhị phân tuyệt đối (byte-identical duplicates)."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def clean_and_standardize_dataset(
    raw_root_dir: Path,
    output_dir: Path,
    target_acne_ids: list,
    prefix: str = "sample"
):
    """
    Tiền xử lý tập ngoại suy theo chuẩn y sinh:
    1. Lọc bỏ các đa giác nếp nhăn (tránh Label Poisoning).
    2. Chuyển tất cả đa giác mụn về class 0.
    3. Loại bỏ ảnh không chứa tổn thương mụn (đảm bảo lesion-positive).
    4. Khử trùng lặp ảnh (Duplicate removal).
    """
    out_img_dir = output_dir / "images"
    out_lbl_dir = output_dir / "labels"

    # Dọn dẹp thư mục cũ để bảo đảm tính reproducible
    if output_dir.exists():
        shutil.rmtree(output_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    seen_hashes = set()

    total_scanned = 0
    saved_count = 0
    skipped_no_acne = 0
    skipped_duplicate = 0
    skipped_no_label = 0

    # Lấy danh sách tất cả các ảnh trong thư mục thô (bao gồm cả train, val, test)
    all_images = sorted([
        p for p in raw_root_dir.rglob("*")
        if p.suffix.lower() in img_extensions and "labels" not in p.parts
    ])
    total_scanned = len(all_images)
    print(f"\n==================================================")
    print(f"BẮT ĐẦU XỬ LÝ: {raw_root_dir.name}")
    print(f"Tổng số tệp ảnh phát hiện: {total_scanned}")

    for img_p in all_images:
        # Tìm file label tương ứng
        lbl_p = None
        if "images" in img_p.parts:
            parts = list(img_p.parts)
            parts[parts.index("images")] = "labels"
            candidate = Path(*parts).with_suffix(".txt")
            if candidate.exists():
                lbl_p = candidate
                
        if lbl_p is None:
            candidate = img_p.with_suffix(".txt")
            if candidate.exists():
                lbl_p = candidate

        if lbl_p is None or not lbl_p.exists():
            skipped_no_label += 1
            continue

        with open(lbl_p, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if not lines:
            skipped_no_label += 1
            continue

        # Kiểm tra trùng lặp ảnh
        img_hash = get_file_md5(img_p)
        if img_hash in seen_hashes:
            skipped_duplicate += 1
            continue

        # LỌC DUY NHẤT CÁC ĐA GIÁC LÀ MỤN
        clean_acne_polygons = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:  # Định dạng polygon YOLOv8/v11/v26 segmentation
                cls_id = int(parts[0])
                if cls_id in target_acne_ids:
                    # Đưa toàn bộ về đơn lớp class 0
                    parts[0] = "0"
                    clean_acne_polygons.append(" ".join(parts))

        # Nếu ảnh này chỉ chứa nếp nhăn (không có mụn) -> bỏ qua để giữ tập lesion-positive
        if not clean_acne_polygons:
            skipped_no_acne += 1
            continue

        seen_hashes.add(img_hash)
        saved_count += 1

        # Đặt tên chuẩn hóa liên tục
        new_name = f"{prefix}_{saved_count:05d}"
        dest_img = out_img_dir / f"{new_name}{img_p.suffix.lower()}"
        dest_lbl = out_lbl_dir / f"{new_name}.txt"

        shutil.copy2(img_p, dest_img)
        with open(dest_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(clean_acne_polygons) + "\n")

    print(f"--- KẾT THÚC TIỀN XỬ LÝ: {output_dir.name} ---")
    print(f"  * Tổng số ảnh quét: {total_scanned}")
    print(f"  * Loại bỏ do chỉ có nếp nhăn (không có mụn): {skipped_no_acne}")
    print(f"  * Loại bỏ do trùng lặp (MD5 hash): {skipped_duplicate}")
    print(f"  * Loại bỏ do không có nhãn hoặc rỗng: {skipped_no_label}")
    print(f"  * SỐ ẢNH MỤN HỢP LỆ CUỐI CÙNG: {saved_count}")
    return saved_count

print("Đã biên dịch hàm tiền xử lý dữ liệu.")

# %%
# Xác định thư mục đầu ra
clean_wrinkles_dir = Path("clean_external_wrinkles")
clean_whitehead_dir = Path("clean_external_whitehead")

# 1. Tiền xử lý bộ Wrinkles: Bỏ nếp nhăn, chỉ giữ mụn
count_wrinkles = clean_and_standardize_dataset(
    raw_root_dir=wrinkles_yaml_path.parent,
    output_dir=clean_wrinkles_dir,
    target_acne_ids=wrinkle_acne_ids,
    prefix="wrinkle_acne"
)

# 2. Tiền xử lý bộ Whitehead: Gom toàn bộ dạng mụn về class 0
count_whitehead = clean_and_standardize_dataset(
    raw_root_dir=whitehead_yaml_path.parent,
    output_dir=clean_whitehead_dir,
    target_acne_ids=whitehead_acne_ids,
    prefix="whitehead_acne"
)

print("\n=> TIỀN XỬ LÝ HOÀN TẤT CHO CẢ 2 BỘ BENCHMARK NGOẠI SUY!")

# %%
def export_dataset_yaml(target_dir: Path, yaml_name: str):
    """Xuất file cấu hình yaml chuẩn Ultralytics để chạy validation khi cần."""
    abs_path = target_dir.resolve()
    yaml_config = {
        "path": str(abs_path).replace("\\", "/"),
        "train": "images",
        "val": "images",
        "test": "images",
        "names": {
            0: "acne"
        }
    }
    yaml_out_path = target_dir / yaml_name
    with open(yaml_out_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_config, f, sort_keys=False)
    print(f"-> Đã tạo file: {yaml_out_path}")

export_dataset_yaml(clean_wrinkles_dir, "clean_wrinkles.yaml")
export_dataset_yaml(clean_whitehead_dir, "clean_whitehead.yaml")

print("\nDữ liệu sạch đã sẵn sàng 100% để chạy Benchmark Inference!")


