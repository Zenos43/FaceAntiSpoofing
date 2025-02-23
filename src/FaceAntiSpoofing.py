import cv2  # Import thư viện OpenCV để xử lý ảnh
import onnxruntime as ort  # Import thư viện ONNX Runtime để chạy mô hình ONNX
import numpy as np  # Import thư viện NumPy để làm việc với mảng
import os  # Import thư viện OS để tương tác với hệ điều hành

# onnx model
class AntiSpoof:
    def __init__(
        self, weights: str = None, model_img_size: int = 128
    ):  # Khởi tạo lớp AntiSpoof
        super().__init__()  # Gọi hàm khởi tạo của lớp cha
        self.weights = weights  # Đường dẫn đến file trọng số của mô hình ONNX
        self.model_img_size = (
            model_img_size  # Kích thước ảnh mà mô hình yêu cầu
        )
        self.ort_session, self.input_name = self._init_session_(
            self.weights
        )  # Khởi tạo phiên ONNX Runtime và lấy tên input của mô hình

    def _init_session_(
        self, onnx_model_path: str
    ):  # Khởi tạo phiên ONNX Runtime
        ort_session = None  # Khởi tạo biến phiên ONNX Runtime
        input_name = None  # Khởi tạo biến tên input của mô hình
        if os.path.isfile(
            onnx_model_path
        ):  # Kiểm tra xem file mô hình ONNX có tồn tại không
            try:
                ort_session = ort.InferenceSession(
                    onnx_model_path, providers=["CUDAExecutionProvider"]
                )  # Thử khởi tạo phiên ONNX Runtime với CUDA
            except:
                ort_session = ort.InferenceSession(
                    onnx_model_path, providers=["CPUExecutionProvider"]
                )  # Nếu không có CUDA, khởi tạo với CPU
            input_name = (
                ort_session.get_inputs()[0].name
            )  # Lấy tên input của mô hình
        return (
            ort_session,
            input_name,
        )  # Trả về phiên ONNX Runtime và tên input của mô hình

    def preprocessing(self, img):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        new_size = (
            self.model_img_size
        )  # Lấy kích thước ảnh mà mô hình yêu cầu
        old_size = img.shape[:2]  # old_size is in (height, width) format
        # Lấy kích thước ảnh gốc (chiều cao, chiều rộng)

        ratio = float(new_size) / max(
            old_size
        )  # Tính tỉ lệ giữa kích thước mới và kích thước lớn nhất của ảnh gốc
        scaled_shape = tuple(
            [int(x * ratio) for x in old_size]
        )  # Tính kích thước mới của ảnh sau khi resize

        # new_size should be in (width, height) format
        img = cv2.resize(
            img, (scaled_shape[1], scaled_shape[0])
        )  # Resize ảnh về kích thước mới

        delta_w = new_size - scaled_shape[1]  # Tính độ lệch chiều rộng
        delta_h = new_size - scaled_shape[0]  # Tính độ lệch chiều cao
        top, bottom = delta_h // 2, delta_h - (
            delta_h // 2
        )  # Tính số pixel cần thêm vào phía trên và phía dưới
        left, right = delta_w // 2, delta_w - (
            delta_w // 2
        )  # Tính số pixel cần thêm vào bên trái và bên phải

        img = cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=[
                0,
                0,
                0,
            ],
        )  # Thêm viền đen vào ảnh để đạt kích thước yêu cầu
        img = (
            img.transpose(2, 0, 1).astype(np.float32) / 255.0
        )  # Chuyển đổi ảnh sang định dạng (C, H, W) và chuẩn hóa giá trị pixel về [0, 1]
        img = np.expand_dims(
            img, axis=0
        )  # Thêm một chiều vào ảnh để tạo thành batch size = 1
        return img  # Trả về ảnh đã được tiền xử lý

    def postprocessing(self, prediction):
        softmax = lambda x: np.exp(x) / np.sum(
            np.exp(x)
        )  # Định nghĩa hàm softmax
        pred = softmax(prediction)  # Áp dụng hàm softmax để lấy xác suất
        return pred  # Trả về xác suất
        # return np.argmax(pred)

    def __call__(
        self, imgs: list
    ):  # Hàm gọi lớp, nhận một danh sách ảnh đầu vào
        if not self.ort_session:  # Kiểm tra xem phiên ONNX Runtime đã được khởi tạo chưa
            return False  # Nếu chưa, trả về False

        preds = []  # Khởi tạo danh sách để lưu trữ kết quả dự đoán
        for img in imgs:  # Duyệt qua từng ảnh trong danh sách
            onnx_result = self.ort_session.run(
                [], {self.input_name: self.preprocessing(img)}
            )  # Chạy mô hình ONNX và lấy kết quả
            pred = onnx_result[0]  # Lấy kết quả dự đoán
            pred = self.postprocessing(pred)  # Hậu xử lý kết quả dự đoán
            preds.append(pred)  # Thêm kết quả dự đoán vào danh sách
        return preds  # Trả về danh sách kết quả dự đoán