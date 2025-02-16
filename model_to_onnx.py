import argparse  # Import thư viện argparse để phân tích cú pháp các đối số dòng lệnh
import os  # Import thư viện os để tương tác với hệ điều hành

import onnx  # Import thư viện onnx để làm việc với mô hình ONNX
import onnxsim  # Import thư viện onnxsim để đơn giản hóa mô hình ONNX
import torch  # Import thư viện torch để làm việc với PyTorch

from src.antispoof_pretrained import (
    AntiSpoofPretrained,
)  # Import lớp AntiSpoofPretrained từ module src.antispoof_pretrained
from src.config import (
    PretrainedConfig,
)  # Import lớp PretrainedConfig từ module src.config

providers = [
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]  # Danh sách các provider để chạy mô hình ONNX


def check_onnx_model(model):
    # Hàm kiểm tra tính hợp lệ của mô hình ONNX
    try:
        onnx.checker.check_model(model)  # Kiểm tra mô hình ONNX
    except onnx.checker.ValidationError as e:
        print("Mô hình ONNX không hợp lệ:", e)  # In thông báo lỗi nếu mô hình không hợp lệ
    else:
        print("Mô hình ONNX hợp lệ!")  # In thông báo nếu mô hình hợp lệ


if __name__ == "__main__":
    # Phân tích cú pháp các đối số
    p = argparse.ArgumentParser(
        description="Chuyển đổi trọng số mô hình từ .pth sang .onnx"
    )  # Tạo một trình phân tích cú pháp đối số với mô tả
    p.add_argument(
        "model_path", type=str, help="Đường dẫn đến trọng số mô hình .pth"
    )  # Thêm đối số cho đường dẫn đến trọng số mô hình .pth
    p.add_argument(
        "num_classes",
        type=int,
        default=2,
        help="Số lượng lớp mà mô hình được huấn luyện để dự đoán",
    )  # Thêm đối số cho số lượng lớp, mặc định là 2
    p.add_argument(
        "--onnx_model_path",
        type=str,
        default=None,
        help="Đường dẫn để lưu trọng số mô hình .onnx đã chuyển đổi",
    )  # Thêm đối số cho đường dẫn để lưu mô hình ONNX, mặc định là None
    p.add_argument(
        "--print_summary",
        type=bool,
        default=False,
        help="Có in thông tin mô hình hay không (cần torchsummary)",
    )  # Thêm đối số để in thông tin mô hình, mặc định là False
    args = p.parse_args()  # Phân tích cú pháp các đối số dòng lệnh

    assert os.path.isfile(
        args.model_path
    ), "Không tìm thấy mô hình {}!".format(
        args.model_path
    )  # Kiểm tra xem tệp mô hình có tồn tại hay không
    # 'saved_models/AntiSpoofing_print-replay_128.pth'
    cnf = PretrainedConfig(
        args.model_path, num_classes=args.num_classes
    )  # Tạo một đối tượng PretrainedConfig với đường dẫn mô hình và số lượng lớp

    model = AntiSpoofPretrained(
        cnf
    ).model  # Tạo một đối tượng AntiSpoofPretrained và lấy mô hình
    print(args.model_path, "đã tải thành công")  # In thông báo đã tải mô hình thành công

    if args.print_summary:
        from torchsummary import summary  # Import hàm summary từ torchsummary

        summary(model)  # In thông tin tóm tắt về mô hình

    onnx_model_path = (
        args.onnx_model_path
    )  # Lấy đường dẫn để lưu mô hình ONNX từ đối số
    if onnx_model_path is None:
        onnx_model_path = cnf.model_path.replace(
            ".pth", ".onnx"
        )  # Nếu đường dẫn không được cung cấp, tạo đường dẫn bằng cách thay thế .pth bằng .onnx
    # Lưu mô hình onnx
    model.eval()  # Đặt mô hình vào chế độ đánh giá
    dummy_input = torch.randn(
        1, 3, cnf.input_size, cnf.input_size
    ).to(
        cnf.device
    )  # Tạo một đầu vào giả để xuất mô hình ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        # verbose=False,
        input_names=["input"],
        output_names=["output"],
        export_params=True,
    )  # Xuất mô hình sang định dạng ONNX
    # Tải mô hình onnx
    onnx_model = onnx.load(
        onnx_model_path
    )  # Tải mô hình ONNX đã xuất
    print("\nKiểm tra mô hình đã xuất")  # In thông báo
    check_onnx_model(
        onnx_model
    )  # Kiểm tra tính hợp lệ của mô hình ONNX đã xuất
    # Đơn giản hóa mô hình
    onnx_model, check = onnxsim.simplify(
        onnx_model,
        # dynamic_input_shape=True,
        # input_shapes={'input': list(dummy_input.shape)}
    )  # Đơn giản hóa mô hình ONNX
    print("\nKiểm tra mô hình đã đơn giản hóa")  # In thông báo
    assert check, "Không thể xác thực mô hình ONNX đã đơn giản hóa"  # Kiểm tra xem mô hình đã đơn giản hóa có hợp lệ hay không
    check_onnx_model(
        onnx_model
    )  # Kiểm tra tính hợp lệ của mô hình ONNX đã đơn giản hóa
    # Lưu mô hình đã đơn giản hóa
    onnx.save(
        onnx_model, onnx_model_path
    )  # Lưu mô hình ONNX đã đơn giản hóa

    print("\nPhiên bản IR:", onnx_model.ir_version)  # In phiên bản IR của mô hình ONNX
    print(
        "Mô hình ONNX đã xuất sang:", onnx_model_path
    )  # In đường dẫn đến mô hình ONNX đã xuất
