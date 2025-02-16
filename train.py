from src.config import TrainConfig  # Import lớp TrainConfig từ module src.config
from src.train_main import TrainMain  # Import lớp TrainMain từ module src.train_main
import argparse  # Import thư viện argparse để phân tích cú pháp các đối số dòng lệnh

if __name__ == "__main__":
    # Phân tích cú pháp các đối số
    p = argparse.ArgumentParser(
        description="Huấn luyện mô hình Face-AntiSpoofing"
    )  # Tạo một trình phân tích cú pháp đối số với mô tả
    p.add_argument(
        "--crop_dir",
        type=str,
        default="data128",
        help="Thư mục con chứa ảnh đã crop",
    )  # Thêm đối số cho thư mục chứa ảnh đã crop, mặc định là 'data128'
    p.add_argument(
        "--input_size",
        type=int,
        default=128,
        help="Kích thước ảnh đầu vào cho mô hình",
    )  # Thêm đối số cho kích thước ảnh đầu vào, mặc định là 128
    p.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Số lượng ảnh trong một batch",
    )  # Thêm đối số cho kích thước batch, mặc định là 256
    p.add_argument(
        "--num_classes",
        type=int,
        default=2,
        choices=[2, 3],
        help="2 cho phân loại nhị phân hoặc 3 cho phân loại live-print-replay",
    )  # Thêm đối số cho số lượng lớp, mặc định là 2, chỉ chấp nhận 2 hoặc 3
    p.add_argument(
        "--job_name",
        type=str,
        default="job",
        help="Hậu tố cho tên mô hình được lưu trong thư mục snapshots",
    )  # Thêm đối số cho tên công việc, mặc định là 'job'
    args = p.parse_args()  # Phân tích cú pháp các đối số dòng lệnh

    if (
        args.num_classes == 2
    ):  # Nếu số lượng lớp là 2 (phân loại nhị phân)
        spoof_categories = (
            "binary"
        )  # Đặt danh mục giả mạo thành 'binary'
    elif (
        args.num_classes == 3
    ):  # Nếu số lượng lớp là 3 (phân loại live-print-replay)
        spoof_categories = [
            [0],
            [1, 2, 3],
            [7, 8, 9],
        ]  # Đặt danh mục giả mạo thành danh sách các lớp

    # Tạo cấu hình
    cnf = TrainConfig(
        crop_dir=args.crop_dir,
        input_size=args.input_size,
        batch_size=args.batch_size,
        spoof_categories=spoof_categories,
    )  # Tạo một đối tượng TrainConfig với các đối số đã phân tích cú pháp
    cnf.set_job(
        args.job_name
    )  # Đặt tên công việc cho cấu hình, sử dụng đối số job_name
    print("Thiết bị:", cnf.device)  # In thiết bị đang được sử dụng (CPU hoặc GPU)

    # Huấn luyện
    trainer = TrainMain(
        cnf
    )  # Tạo một đối tượng TrainMain với cấu hình đã tạo
    trainer.train_model()  # Bắt đầu quá trình huấn luyện mô hình
    print("Hoàn thành")  # In thông báo hoàn thành sau khi huấn luyện xong
