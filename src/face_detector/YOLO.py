# Nhập các thư viện cần thiết
import cv2  # Thư viện xử lý ảnh
import onnxruntime as ort  # Thư viện chạy mô hình ONNX
import time  # Thư viện xử lý thời gian
import numpy as np  # Thư viện xử lý mảng số học
from .utils import non_max_suppression, scale_coords, letterbox  # Nhập các hàm hữu ích từ module utils
import os  # Thư viện thao tác với hệ thống tệp tin

# Mô hình YOLOv5
class YOLOv5:
    # Hàm khởi tạo lớp YOLOv5
    def __init__(self,
                 weights: str = None,  # Đường dẫn tới mô hình ONNX
                 input_res: tuple = (640, 640),  # Kích thước đầu vào của hình ảnh
                 batch_size: int = 1):  # Kích thước batch
        super().__init__()
        self.weights = weights  # Lưu đường dẫn mô hình
        self.input_res = input_res  # Lưu kích thước đầu vào
        self.batch_size = batch_size  # Lưu kích thước batch
        self.ort_session, self.input_name = self._init_session_(self.weights)  # Khởi tạo phiên làm việc ONNX
        self.max_detection = 1000  # Giới hạn số lượng đối tượng phát hiện

    # Hàm khởi tạo phiên làm việc ONNX
    def _init_session_(self, path_onnx_model: str):
        ort_session = None  # Biến lưu phiên làm việc ONNX
        input_name = None  # Biến lưu tên của input
        if os.path.isfile(path_onnx_model):  # Kiểm tra xem mô hình có tồn tại không
            try:
                # Khởi tạo phiên làm việc ONNX với GPU nếu có thể
                ort_session = ort.InferenceSession(path_onnx_model, providers=['CUDAExecutionProvider'])
            except:
                # Nếu không sử dụng GPU, sử dụng CPU
                ort_session = ort.InferenceSession(path_onnx_model, providers=['CPUExecutionProvider'])
            input_name = ort_session.get_inputs()[0].name  # Lấy tên input của mô hình
        return ort_session, input_name  # Trả về phiên làm việc và tên input

    # Hàm tiền xử lý hình ảnh
    def preprocessing(self, imgs: list):
        imgs_input = []  # Danh sách lưu trữ các ảnh đã qua tiền xử lý
        for img in imgs:
            # Thực hiện thay đổi kích thước và tạo ảnh có tỷ lệ phù hợp
            img_input, ratio, (dw, dh) = letterbox(img,
                                                   self.input_res,
                                                   auto=False,
                                                   scaleFill=False,
                                                   scaleup=True,
                                                   stride=32)
            img_input = img_input.transpose(2, 0, 1)  # Chuyển ảnh từ HWC (chiều cao, rộng, kênh) sang CHW
            img_input = np.ascontiguousarray(img_input)  # Chuyển ảnh thành mảng liên tục trong bộ nhớ
            img_input = img_input.astype(np.float32)  # Chuyển kiểu dữ liệu của ảnh thành float32
            img_input /= 255.0  # Chuẩn hóa giá trị pixel từ 0-255 thành 0-1
            img_input = np.expand_dims(img_input, axis=0)  # Thêm một chiều batch vào ảnh
            imgs_input.append(img_input)  # Thêm ảnh đã tiền xử lý vào danh sách
        return imgs_input  # Trả về danh sách các ảnh đã qua tiền xử lý

    # Hàm hậu xử lý kết quả phát hiện
    def postprocessing(self, prediction_bboxes, imgs, conf_thresh=0.25, iou_thresh=0.1, max_detection=1):
        assert len(prediction_bboxes) == len(imgs), f"Size prediction {len(prediction_bboxes)} not equal size images {len(imgs)}"
        # Áp dụng non-max suppression (NMS) để loại bỏ các bounding box trùng lặp
        pred = non_max_suppression(prediction_bboxes,
                                   conf_thresh=conf_thresh,
                                   iou_thresh=iou_thresh,
                                   max_det=max_detection)
        for i, det in enumerate(pred):  # Lặp qua các phát hiện cho mỗi ảnh
            if len(det):
                # Quy đổi tọa độ bounding box từ kích thước ảnh đầu vào sang kích thước ảnh gốc
                det[:, :4] = scale_coords(self.input_res, det[:, :4], imgs[i].shape).round()
        return pred  # Trả về kết quả đã hậu xử lý

    # Hàm gọi mô hình để dự đoán
    def __call__(self, imgs, conf_thresh=0.25, iou_thresh=0.45, max_detection=1):
        if not self.ort_session:  # Kiểm tra xem phiên làm việc ONNX đã được khởi tạo chưa
            return False

        if self.batch_size == 1:  # Nếu batch_size là 1, dự đoán cho từng ảnh một
            preds = []  # Danh sách lưu trữ kết quả dự đoán
            for img in imgs:
                # Chạy mô hình ONNX để dự đoán
                onnx_result = self.ort_session.run([],
                                                   {self.input_name: self.preprocessing([img])[0]})
                pred = onnx_result[0]  # Lấy kết quả dự đoán
                # Áp dụng hậu xử lý lên kết quả dự đoán
                pred = self.postprocessing(prediction_bboxes=pred,
                                           imgs=[img],
                                           conf_thresh=conf_thresh,
                                           iou_thresh=iou_thresh,
                                           max_detection=max_detection)
                preds.append(pred[0])  # Thêm kết quả vào danh sách
            return preds  # Trả về kết quả dự đoán cho từng ảnh

        else:  # Nếu batch_size lớn hơn 1, dự đoán cho tất cả ảnh cùng lúc
            input_imgs = self.preprocessing(imgs)  # Tiền xử lý tất cả ảnh
            input_imgs = np.concatenate(input_imgs, axis=0)  # Nối các ảnh lại thành một mảng duy nhất
            # Chạy mô hình ONNX để dự đoán cho tất cả ảnh
            onnx_result = self.ort_session.run([], {self.input_name: input_imgs})
            pred = onnx_result[0]  # Lấy kết quả dự đoán
            # Áp dụng hậu xử lý lên kết quả dự đoán
            pred = self.postprocessing(prediction_bboxes=pred,
                                       imgs=imgs,
                                       conf_thresh=conf_thresh,
                                       iou_thresh=iou_thresh,
                                       max_detection=max_detection)
        return pred  # Trả về kết quả dự đoán cho tất cả ảnh