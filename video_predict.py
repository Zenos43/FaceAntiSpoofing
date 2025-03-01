import argparse  # Import thư viện argparse để phân tích cú pháp các đối số dòng lệnh

import cv2  # Import thư viện cv2 cho các tác vụ xử lý ảnh
import numpy as np  # Import thư viện numpy cho các phép toán số học

from src.face_detector import YOLOv5  # Import lớp YOLOv5 để phát hiện khuôn mặt
from src.FaceAntiSpoofing import AntiSpoof  # Import lớp AntiSpoof để chống giả mạo khuôn mặt

COLOR_REAL = (0, 255, 0)  # Định nghĩa màu cho khuôn mặt thật (Xanh lá)
COLOR_FAKE = (0, 0, 255)  # Định nghĩa màu cho khuôn mặt giả (Đỏ)
COLOR_UNKNOWN = (
    127,
    127,
    127,
)  # Định nghĩa màu cho khuôn mặt không xác định (Xám)


def increased_crop(img, bbox: tuple, bbox_inc: float = 1.5):
    # Cắt khuôn mặt dựa trên bounding box của nó
    real_h, real_w = img.shape[:2]  # Lấy chiều cao và chiều rộng của ảnh

    x, y, w, h = bbox  # Giải nén tọa độ bounding box
    w, h = w - x, h - y  # Tính chiều rộng và chiều cao của bounding box
    l = max(w, h)  # Tìm giá trị lớn nhất giữa chiều rộng và chiều cao

    xc, yc = x + w / 2, y + h / 2  # Tính tâm của bounding box
    x, y = int(xc - l * bbox_inc / 2), int(
        yc - l * bbox_inc / 2
    )  # Tính tọa độ góc trên bên trái mới
    x1 = 0 if x < 0 else x  # Đảm bảo x1 không âm
    y1 = 0 if y < 0 else y  # Đảm bảo y1 không âm
    x2 = (
        real_w if x + l * bbox_inc > real_w else x + int(l * bbox_inc)
    )  # Đảm bảo x2 không vượt quá giới hạn
    y2 = (
        real_h if y + l * bbox_inc > real_h else y + int(l * bbox_inc)
    )  # Đảm bảo y2 không vượt quá giới hạn

    img = img[y1:y2, x1:x2, :]  # Cắt ảnh
    img = cv2.copyMakeBorder(
        img,
        y1 - y,
        int(l * bbox_inc - y2 + y),
        x1 - x,
        int(l * bbox_inc) - x2 + x,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )  # Thêm padding vào ảnh
    return img  # Trả về ảnh đã cắt và padding


def make_prediction(img, face_detector, anti_spoof):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển đổi ảnh sang RGB
    bbox_list = face_detector([img])  # Lấy danh sách bounding box

    # Kiểm tra xem bbox_list có rỗng hay không
    if not bbox_list or not isinstance(bbox_list[0], np.ndarray):
        return None  # Trả về None nếu không tìm thấy bounding box nào

    bbox = bbox_list[0]  # Lấy bounding box đầu tiên
    if bbox.shape[0] == 0:
        return None  # Trả về None nếu bounding box rỗng

    bbox = bbox.flatten()[:4].astype(int)  # Làm phẳng và chuyển đổi sang số nguyên

    pred = anti_spoof([increased_crop(img, bbox, bbox_inc=1.5)])[
        0
    ]  # Thực hiện dự đoán chống giả mạo
    score = pred[0][0]  # Lấy điểm số
    label = np.argmax(pred)  # Lấy nhãn

    return bbox, label, score  # Trả về bounding box, nhãn và điểm số


if __name__ == "__main__":
    # Phân tích cú pháp các đối số
    def check_zero_to_one(value):
        fvalue = float(value)
        if fvalue <= 0 or fvalue >= 1:
            raise argparse.ArgumentTypeError("%s là một giá trị không hợp lệ" % value)
        return fvalue

    p = argparse.ArgumentParser(
        description="Phát hiện tấn công giả mạo trên luồng video"
    )  # Tạo một trình phân tích cú pháp đối số
    p.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Đường dẫn đến video để dự đoán",
    )  # Thêm một đối số cho đường dẫn video đầu vào
    p.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Đường dẫn để lưu video đã xử lý",
    )  # Thêm một đối số cho đường dẫn video đầu ra
    p.add_argument(
        "--model_path",
        "-m",
        type=str,
        default="saved_models/AntiSpoofing_bin_1.5_128.onnx",
        help="Đường dẫn đến mô hình ONNX",
    )  # Thêm một đối số cho đường dẫn mô hình
    p.add_argument(
        "--threshold",
        "-t",
        type=check_zero_to_one,
        default=0.5,
        help="Ngưỡng xác suất khuôn mặt thật trên đó dự đoán được coi là đúng",
    )  # Thêm một đối số cho ngưỡng
    args = p.parse_args()  # Phân tích cú pháp các đối số

    face_detector = YOLOv5(
        "saved_models/yolov5s-face.onnx"
    )  # Tạo một đối tượng phát hiện khuôn mặt YOLOv5
    anti_spoof = AntiSpoof(args.model_path)  # Tạo một đối tượng AntiSpoof

    # Tạo một đối tượng video capture
    if args.input:  # file
        vid_capture = cv2.VideoCapture(args.input)  # Mở tệp video đầu vào
    else:  # webcam
        vid_capture = cv2.VideoCapture(
            0, cv2.CAP_DSHOW
        )  # Mở webcam mặc định

    frame_width = int(vid_capture.get(3))  # Lấy chiều rộng của khung hình video
    frame_height = int(
        vid_capture.get(4)
    )  # Lấy chiều cao của khung hình video
    frame_size = (frame_width, frame_height)  # Xác định kích thước khung hình
    print("Kích thước khung hình  :", frame_size)  # In kích thước khung hình

    if not vid_capture.isOpened():
        print("Lỗi khi mở luồng video")  # In một thông báo lỗi
        fps = 24  # Đặt tốc độ khung hình mặc định
    else:
        fps = vid_capture.get(5)  # Lấy thông tin về tốc độ khung hình
        print("Tốc độ khung hình  : ", fps, "FPS")  # In tốc độ khung hình
        if fps == 0:
            fps = 24  # Đặt tốc độ khung hình mặc định

    # videowriter
    output = None  # Khởi tạo output thành None
    if args.output:
        output = cv2.VideoWriter(
            args.output, cv2.VideoWriter_fourcc(*"XVID"), fps, frame_size
        )  # Tạo một đối tượng video writer
    print("Video đang được xử lý. Nhấn 'Q' hoặc 'Esc' để thoát")  # In một thông báo

    # Xử lý khung hình
    rec_width = max(
        1, int(frame_width / 240)
    )  # Tính chiều rộng hình chữ nhật
    txt_offset = int(frame_height / 50)  # Tính độ lệch văn bản
    txt_width = max(1, int(frame_width / 480))  # Tính chiều rộng văn bản
    while vid_capture.isOpened():
        ret, frame = vid_capture.read()  # Đọc một khung hình từ video
        if ret:
            # Dự đoán điểm số của khuôn mặt thật
            pred = make_prediction(
                frame, face_detector, anti_spoof
            )  # Thực hiện dự đoán
            # Nếu phát hiện khuôn mặt
            if pred is not None:
                (x1, y1, x2, y2), label, score = pred  # Giải nén kết quả
                if label == 0:
                    if score > args.threshold:
                        res_text = "REAL      {:.2f}".format(
                            score
                        )  # Định dạng văn bản kết quả
                        color = COLOR_REAL  # Đặt màu thành xanh lá
                    else:
                        res_text = "UNKNOWN"  # Đặt văn bản kết quả thành không xác định
                        color = COLOR_UNKNOWN  # Đặt màu thành xám
                else:
                    res_text = "FAKE      {:.2f}".format(
                        score
                    )  # Định dạng văn bản kết quả
                    color = COLOR_FAKE  # Đặt màu thành đỏ

                # Vẽ bounding box với nhãn
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), color, rec_width
                )  # Vẽ một hình chữ nhật xung quanh khuôn mặt
                cv2.putText(
                    frame,
                    res_text,
                    (x1, y1 - txt_offset),
                    cv2.FONT_HERSHEY_COMPLEX,
                    (x2 - x1) / 250,
                    color,
                    txt_width,
                )  # Đặt văn bản kết quả lên khung hình

            if output is not None:
                output.write(frame)  # Ghi khung hình vào video đầu ra

            # Nếu video được chụp từ webcam
            if not args.input:
                cv2.imshow("DNTU_FaceAntiSpoofing", frame)  # Hiển thị khung hình
                key = cv2.waitKey(20)  # Chờ một phím được nhấn
                if (key == ord("q")) or key == 27:
                    break  # Thoát khỏi vòng lặp nếu 'q' hoặc 'Esc' được nhấn
        else:
            print("Tắt tiến trình")  # In một thông báo
            break  # Thoát khỏi vòng lặp

    # Giải phóng các đối tượng video capture và writer
    vid_capture.release()  # Giải phóng đối tượng video capture
    if output is not None:
        output.release()  # Giải phóng đối tượng video writer
