from collections import OrderedDict
import torch
import torch.nn.functional as F

from src.NN import MultiFTNet, MiniFASNetV2SE

class AntiSpoofPretrained:
    def __init__(self, cnf):
        # Khởi tạo các thông số từ cấu hình (cnf)
        self.device = cnf.device  # Thiết bị (CPU hoặc GPU)
        self.input_size = cnf.input_size  # Kích thước đầu vào của ảnh
        self.kernel_size = cnf.kernel_size  # Kích thước của kernel
        self.num_classes = cnf.num_classes  # Số lớp phân loại
        self.model = MiniFASNetV2SE(conv6_kernel=self.kernel_size, 
                                    num_classes=self.num_classes).to(self.device)  # Khởi tạo mô hình
        self.model_path = cnf.model_path  # Đường dẫn tới mô hình đã huấn luyện
        
        # Tải trọng số của mô hình đã huấn luyện
        state_dict = torch.load(self.model_path, map_location=self.device)
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if not 'FTGenerator' in key:  # Bỏ qua các tham số có chứa 'FTGenerator'
                name_key = key.replace('module.model.', '', 1)  # Xóa prefix 'module.model.' trong tên key
                new_state_dict[name_key] = value
        # Nạp trọng số vào mô hình
        self.model.load_state_dict(new_state_dict)
        
    def predict(self, img):
        img = img.to(self.device)  # Di chuyển ảnh vào thiết bị
        self.model.eval()  # Chuyển mô hình sang chế độ đánh giá (evaluation mode)
        with torch.no_grad():  # Tắt tính toán gradient (không cần thiết khi dự đoán)
            result = self.model.forward(img)  # Tiến hành dự đoán
            result = F.softmax(result, -1).cpu().numpy()  # Áp dụng softmax và chuyển kết quả về numpy
        return result  # Trả về kết quả dự đoán
    
class AntiSpoofPretrainedFT(AntiSpoofPretrained):
    def __init__(self, cnf):
        # Khởi tạo các thông số từ cấu hình (cnf)
        self.device = cnf.device  # Thiết bị (CPU hoặc GPU)
        self.input_size = cnf.input_size  # Kích thước đầu vào của ảnh
        self.kernel_size = cnf.kernel_size  # Kích thước của kernel
        self.num_classes = cnf.num_classes  # Số lớp phân loại
        self.model = MultiFTNet(conv6_kernel=self.kernel_size, 
                                num_classes=self.num_classes).to(self.device)  # Khởi tạo mô hình
        self.model_path = cnf.model_path  # Đường dẫn tới mô hình đã huấn luyện
        
        # Tải trọng số của mô hình đã huấn luyện
        state_dict = torch.load(self.model_path, map_location=self.device)
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            name_key = key.replace('module.', '', 1)  # Xóa prefix 'module.' trong tên key
            new_state_dict[name_key] = value
        # Nạp trọng số vào mô hình
        self.model.load_state_dict(new_state_dict)