# Original code https://github.com/minivision-ai/Silent-Face-Anti-Spoofing by @zhuyingSeu
# Modified by Zenos43
# Validation and training loaders implemented
# CelebADataset with and w/o FT implemented

# Thư viện hệ thống để tương tác với hệ điều hành
import os
# Thư viện OpenCV để xử lý ảnh
import cv2
# Thư viện PyTorch để xây dựng và huấn luyện mạng nơ-ron
import torch
# Các lớp DataLoader và Dataset từ PyTorch để quản lý dữ liệu
from torch.utils.data import DataLoader, Dataset
# Hàm train_test_split từ scikit-learn để chia dữ liệu thành tập huấn luyện và tập kiểm tra
from sklearn.model_selection import train_test_split
# Các phép biến đổi ảnh từ torchvision
import torchvision.transforms as T
# Các hàm chức năng biến đổi ảnh từ torchvision
import torchvision.transforms.functional as F
# Thư viện NumPy để làm việc với mảng và các phép toán số học
import numpy as np
# Thư viện Pandas để làm việc với dữ liệu dạng bảng
import pandas as pd


# Hàm để đọc ảnh bằng OpenCV
def opencv_loader(path):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(path)
    # Chuyển đổi không gian màu từ BGR (mặc định của OpenCV) sang RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Hàm để tạo ảnh Fourier Transform (FT) từ ảnh đầu vào
def generate_FT(image):
    # Chuyển đổi ảnh màu sang ảnh xám
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Tính toán biến đổi Fourier 2D
    f = np.fft.fft2(image)
    # Dịch chuyển các thành phần tần số về trung tâm
    fshift = np.fft.fftshift(f)
    # Tính toán độ lớn của các thành phần tần số và lấy logarit
    fimg = np.log(np.abs(fshift)+1)
    # Tìm giá trị lớn nhất và nhỏ nhất trong ảnh FT
    maxx = -1
    minn = 100000
    for i in range(len(fimg)):
        if maxx < max(fimg[i]):
            maxx = max(fimg[i])
        if minn > min(fimg[i]):
            minn = min(fimg[i])
    # Chuẩn hóa ảnh FT về khoảng [0, 1]
    fimg = (fimg - minn+1) / (maxx - minn+1)
    return fimg

# Lớp Dataset tùy chỉnh cho tập dữ liệu CelebA
class CelebADataset(Dataset):
    # Hàm khởi tạo
    def __init__(self, root, labels, transform=None, target_transform=None,
                 loader=opencv_loader):
        # Đường dẫn gốc của tập dữ liệu
        self.root = root
        # Bảng chứa thông tin về nhãn của dữ liệu
        self.labels = labels
        # Các phép biến đổi ảnh
        self.transform = transform
        # Các phép biến đổi nhãn
        self.target_transform = target_transform
        # Hàm đọc ảnh
        self.loader = loader

    # Hàm trả về kích thước của tập dữ liệu
    def __len__(self):
        return len(self.labels)
    
    # Hàm trả về một mẫu dữ liệu tại vị trí index
    def __getitem__(self, idx):
        # Tạo đường dẫn đầy đủ đến ảnh
        path = os.path.join(self.root, self.labels.iloc[idx, 0])
        # Đọc ảnh
        sample = self.loader(path)
        # Lấy nhãn của ảnh
        target = self.labels.iloc[idx, 1]
        
        # Kiểm tra xem ảnh có bị lỗi không
        if sample is None:
            print('image is None --> ', path)
        assert sample is not None
        
        # Áp dụng các phép biến đổi ảnh
        if self.transform is not None:
            try:
                sample = self.transform(sample)
            except Exception as err:
                print('Error Occured: %s' % err, path)
        # Áp dụng các phép biến đổi nhãn
        if self.target_transform is not None:
            target = self.target_transform(target)
        # Trả về ảnh và nhãn
        return sample, target


# Lớp Dataset tùy chỉnh cho tập dữ liệu CelebA với ảnh FT
class CelebADatasetFT(CelebADataset):
    # Hàm khởi tạo
    def __init__(self, root, labels, transform=None, target_transform=None,
                 loader=opencv_loader, ft_size=(10,10)):
        # Gọi hàm khởi tạo của lớp cha
        super().__init__(root, labels, transform, 
                         target_transform, loader)
        # Kích thước của ảnh FT
        self.ft_size = ft_size
    
    # Hàm trả về một mẫu dữ liệu tại vị trí index
    def __getitem__(self, idx):
        # Tạo đường dẫn đầy đủ đến ảnh
        path = os.path.join(self.root, self.labels.iloc[idx, 0])
        # Đọc ảnh
        sample = self.loader(path)
        # Lấy nhãn của ảnh
        target = self.labels.iloc[idx, 1]
        
        # Tạo ảnh FT từ ảnh gốc
        ft_sample = generate_FT(sample)
        # Kiểm tra xem ảnh có bị lỗi không
        if sample is None:
            print('image is None --> ', path)
        if ft_sample is None:
            print('FT image is None --> ', path)
        assert sample is not None

        # Thay đổi kích thước ảnh FT
        ft_sample = cv2.resize(ft_sample, self.ft_size)
        # Chuyển đổi ảnh FT sang tensor PyTorch
        ft_sample = torch.from_numpy(ft_sample).float()
        # Thêm một chiều vào tensor ảnh FT
        ft_sample = torch.unsqueeze(ft_sample, 0)

        # Áp dụng các phép biến đổi ảnh
        if self.transform is not None:
            try:
                sample = self.transform(sample)
            except Exception as err:
                print('Error occured: %s' % err, path)
        # Áp dụng các phép biến đổi nhãn
        if self.target_transform is not None:
            target = self.target_transform(target)
        # Trả về ảnh gốc, ảnh FT và nhãn
        return sample, ft_sample, target


# Lớp để thêm padding vuông vào ảnh
class SquarePad:
    # Hàm gọi lớp như một hàm
    def __call__(self, image):
        # Tìm kích thước lớn nhất của ảnh
        max_wh = max(image.size)
        # Tính toán lượng padding cần thiết ở mỗi bên
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        # Tính toán lượng padding cần thiết ở phía đối diện
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        # Tạo tuple chứa thông tin về padding
        padding = (p_left, p_top, p_right, p_bottom)
        # Thêm padding vào ảnh
        return F.pad(image, padding, 0, 'constant')
    
# Hàm để biến đổi nhãn
def transform_labels(labels, categories):
    # Nếu chỉ có hai loại nhãn (ví dụ: thật/giả)
    if categories == 'binary':
        # Biến đổi nhãn thành 0 hoặc 1
        spoof_transform = lambda t: 0 if t == 0 else 1
    # Nếu có nhiều loại nhãn
    else:
        # Tìm vị trí của nhãn trong danh sách các loại nhãn
        spoof_transform = lambda t: next(i for i, l in enumerate(categories) if t in l)
    return labels.apply(spoof_transform)


# Hàm để tạo DataLoader cho tập huấn luyện và tập kiểm tra
def get_train_valid_loader(cnf):
    
    # Các phép biến đổi ảnh cho tập huấn luyện
    train_transform = T.Compose([
        T.ToPILImage(), # Chuyển đổi ảnh sang định dạng PIL
        #SquarePad(),
        T.Resize((cnf.input_size, cnf.input_size)), # Thay đổi kích thước ảnh
        T.RandomResizedCrop(size=tuple(2*[cnf.input_size]), scale=(0.9, 1.1)), # Cắt ngẫu nhiên một phần ảnh và thay đổi kích thước
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1), # Thay đổi độ sáng, độ tương phản, độ bão hòa và màu sắc của ảnh
        T.RandomRotation(10), # Xoay ảnh ngẫu nhiên
        T.RandomHorizontalFlip(), # Lật ảnh theo chiều ngang ngẫu nhiên
        T.ToTensor() # Chuyển đổi ảnh sang tensor PyTorch
    ])
    
    # Các phép biến đổi ảnh cho tập kiểm tra
    valid_transform = T.Compose([
        T.ToPILImage(), # Chuyển đổi ảnh sang định dạng PIL
        #SquarePad(),
        T.Resize((cnf.input_size, cnf.input_size)), # Thay đổi kích thước ảnh
        T.ToTensor() # Chuyển đổi ảnh sang tensor PyTorch
    ])
    
    # Đọc thông tin về nhãn từ file CSV
    train_labels = pd.read_csv(cnf.labels_path)
    
    # Biến đổi nhãn nếu cần thiết
    if cnf.spoof_categories is not None:
        train_labels.iloc[:,1] = transform_labels(train_labels.iloc[:,1],
                                                  cnf.spoof_categories)
    # Cân bằng dữ liệu nếu cần thiết
    if cnf.class_balancing is not None:
        cb = cnf.class_balancing
        # Giảm số lượng mẫu của các lớp có số lượng mẫu lớn hơn
        if cb == 'down':
            value_counts = train_labels.iloc[:,1].value_counts()
            train_downsampled = [
                train_labels[train_labels.iloc[:,1]==value_counts.index[-1]]]
            for value in value_counts.index[:-1]:
                train_downsampled.append(
                    train_labels[train_labels.iloc[:,1]==value].sample(
                        value_counts.min()))
            train_labels = pd.concat(train_downsampled)
    
    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    train_labels, valid_labels = train_test_split(train_labels, 
                                                  test_size=cnf.valid_size, 
                                                  random_state=20220826)
    
    # Reset index của các DataFrame
    train_labels = train_labels.reset_index(drop=True)
    valid_labels = valid_labels.reset_index(drop=True)
    
    # Tạo DataLoader cho tập huấn luyện
    train_loader = DataLoader(
        CelebADatasetFT(cnf.train_path, train_labels, train_transform, 
                        None, ft_size=cnf.ft_size), 
        batch_size=cnf.batch_size, # Kích thước batch
        shuffle=True, pin_memory=True, # Trộn dữ liệu, sử dụng bộ nhớ ghim
    )
    # Tạo DataLoader cho tập kiểm tra
    valid_loader = DataLoader(
        CelebADataset(cnf.train_path, valid_labels, valid_transform, None), 
        batch_size=cnf.batch_size, # Kích thước batch
        shuffle=True, pin_memory=True, # Trộn dữ liệu, sử dụng bộ nhớ ghim
    )
    
    # Trả về DataLoader cho tập huấn luyện và tập kiểm tra
    return train_loader, valid_loader


# Hàm để tạo DataLoader cho tập kiểm thử
def get_test_loader(cnf):
    
    # Các phép biến đổi ảnh cho tập kiểm thử
    test_transform = T.Compose([
        T.ToPILImage(), # Chuyển đổi ảnh sang định dạng PIL
        SquarePad(), # Thêm padding vuông
        T.Resize(size = cnf.input_size), # Thay đổi kích thước ảnh
        T.ToTensor() # Chuyển đổi ảnh sang tensor PyTorch
    ])
    
    # Đọc thông tin về nhãn từ file CSV
    test_labels = pd.read_csv(cnf.labels_path)
    
    # Biến đổi nhãn nếu cần thiết
    if cnf.spoof_categories is not None:
        test_labels.iloc[:,1] = transform_labels(test_labels.iloc[:,1],
                                                 cnf.spoof_categories)
    # Tạo DataLoader cho tập kiểm thử
    test_loader = DataLoader(
        CelebADataset(cnf.test_path, test_labels, test_transform, None), 
        batch_size=cnf.batch_size, pin_memory=True, # Kích thước batch, sử dụng bộ nhớ ghim
    )
    
    # Trả về DataLoader cho tập kiểm thử
    return test_loader
```