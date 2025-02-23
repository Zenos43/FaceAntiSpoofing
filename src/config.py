import os
import torch
from datetime import datetime

SNAPSHOT_PATH = './logs/snapshot'  # Đường dẫn lưu các snapshot
LOG_PATH = './logs/jobs'           # Đường dẫn lưu các log
DATA_PATH = './CelebA_Spoof_crop'  # Đường dẫn tới dữ liệu CelebA_Spoof

class CelebAattr(object):
    # Các thuộc tính khuôn mặt từ chỉ số 0 - 39
    FACE_ATTR = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
        'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
        'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
        'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
        'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
        'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
    ]
    # Chỉ số 40
    SPOOF_TYPE = ['Live',                               # 0     - sống
        'Photo', 'Poster', 'A4',                        # 1,2,3 - IN ẢNH
        'Face Mask', 'Upper Body Mask', 'Region Mask',  # 4,5,6 - MẶT NẠ GIẤY
        'PC', 'Pad', 'Phone',                           # 7,8,9 - PHÁT LẠI
        '3D Mask'                                       # 10    - MẶT NẠ 3D
    ]
    # Chỉ số 41
    ILLUMINATION = ['Live', 'Normal', 'Strong', 'Back', 'Dark']  # Các loại ánh sáng
    # Chỉ số 42
    ENVIRONMENT = ['Live', 'Indoor', 'Outdoor']  # Môi trường (Sống, Trong nhà, Ngoài trời)


def get_num_classes(spoof_categories):
    '''
    `0    ` : live (sống)
    `1,2,3` : PRINT (in ảnh)
    `4,5,6` : PAPER CUT (mặt nạ giấy)
    `7,8,9` : REPLAY (phát lại)
    `10   ` : 3D MASK (mặt nạ 3D)
    '''
    
    if spoof_categories == 'binary':  # Nếu là phân loại nhị phân
        num_classes = 2
    else:
        assert isinstance(spoof_categories, list), "spoof_categories phải là danh sách, nhưng nhận được {}".format(spoof_categories)
        num_classes = len(spoof_categories)  # Số lớp phân loại
    return num_classes

def get_kernel(height, width):
    kernel_size = ((height + 15) // 16, (width + 15) // 16)  # Kích thước kernel (lọc)
    return kernel_size

class TrainConfig(object):
    def __init__(self, input_size=128, batch_size=256, 
                 spoof_categories='binary', class_balancing=None, 
                 crop_dir='data128'):
        # Các tham số huấn luyện
        self.lr = 1e-1  # Học tỷ lệ
        self.milestones = [10, 15, 22, 30]  # Các mốc giảm tỷ lệ học
        self.gamma = 0.1  # Hệ số giảm dần
        self.epochs = 50  # Số vòng huấn luyện
        self.momentum = 0.9  # Động lực
        self.batch_size = batch_size  # Kích thước batch
        self.valid_size = 0.2  # Kích thước tập hợp kiểm tra
        self.class_balancing = class_balancing  # Cân bằng lớp
        
        # Dữ liệu
        self.input_size = input_size  # Kích thước đầu vào
        self.train_path = '{}/{}/train'.format(DATA_PATH, crop_dir)  # Đường dẫn dữ liệu huấn luyện
        self.labels_path = '{}/{}/train/train_target.csv'.format(DATA_PATH, crop_dir)  # Đường dẫn nhãn huấn luyện
        self.spoof_categories = spoof_categories  # Các loại giả mạo

        # Mô hình
        self.num_classes = get_num_classes(spoof_categories)  # Số lớp phân loại
        self.input_channel = 3  # Số kênh đầu vào
        self.embedding_size = 128  # Kích thước embedding
        self.kernel_size = get_kernel(input_size, input_size)  # Kích thước kernel
        # Kích thước Fourier của ảnh
        self.ft_size = [2*s for s in self.kernel_size]
        
        # TensorBoard
        self.board_loss_per_epoch = 10  # Số lần ghi loss trong mỗi epoch

    def set_job(self, name, device_id=0):
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')  # Thời gian hiện tại

        self.device = "cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu"  # Thiết bị sử dụng (GPU nếu có)

        self.job_dir = "AntiSpoofing_{}".format(self.input_size)  # Thư mục công việc
        self.job_name = "AntiSpoofing_{}_{}".format(name, self.input_size)  # Tên công việc
        # Đường dẫn log
        self.log_path = "{}/{}/{}_{}".format(
            LOG_PATH, self.job_dir, name, current_time)
        if not os.path.exists(self.log_path):  # Nếu thư mục log không tồn tại thì tạo mới
            os.makedirs(self.log_path)
        
        # Đường dẫn lưu mô hình
        self.model_path = '{}/{}/{}_{}'.format(
            SNAPSHOT_PATH, self.job_dir, name, current_time)
        if not os.path.exists(self.model_path):  # Nếu thư mục lưu mô hình không tồn tại thì tạo mới
            os.makedirs(self.model_path)


class PretrainedConfig(object):
    def __init__(self, model_path, device_id=0, input_size=128, num_classes=2):
        self.model_path = model_path  # Đường dẫn mô hình đã huấn luyện trước
        self.device = "cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu"  # Thiết bị sử dụng
        self.input_size = input_size  # Kích thước đầu vào
        self.kernel_size = get_kernel(input_size, input_size)  # Kích thước kernel
        self.num_classes = num_classes  # Số lớp phân loại


class TestConfig(PretrainedConfig):
    def __init__(self, model_path, device_id=0, input_size=128, 
                 batch_size=1, spoof_categories='binary', crop_dir='data128'):
        super().__init__(model_path, device_id, input_size, 
                         get_num_classes(spoof_categories))  # Gọi constructor của lớp cha
        self.test_path = '{}/{}/test'.format(DATA_PATH, crop_dir)  # Đường dẫn dữ liệu kiểm tra
        self.labels_path = '{}/{}/test/test_target.csv'.format(DATA_PATH, crop_dir)  # Đường dẫn nhãn kiểm tra
        self.spoof_categories = spoof_categories  # Các loại giả mạo
        self.batch_size = batch_size  # Kích thước batch trong kiểm tra