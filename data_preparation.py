import os  # Import thư viện os để tương tác với hệ điều hành
from tqdm import tqdm as tqdm  # Import tqdm để hiển thị thanh tiến trình
import pandas as pd  # Import thư viện pandas để làm việc với dữ liệu dạng bảng
import argparse  # Import thư viện argparse để phân tích cú pháp các đối số dòng lệnh
import cv2  # Import thư viện cv2 để xử lý ảnh

CELEBA_DIR = "CelebA_Spoof/"  # Đường dẫn đến thư mục chứa dữ liệu gốc CelebA
CROP_DIR = (
    "CelebA_Spoof_crop/"
)  # Đường dẫn đến thư mục để lưu dữ liệu đã crop
# spoof_types = [0, 1, 2, 3, 7, 8, 9] # Spoof атаки, которые оставляем


def read_image(image_path, bbox_inc: float = 1.5):
    """
    Đọc một ảnh từ đường dẫn đầu vào và crop nó với bbox

    params:
        - `image_path` : str - đường dẫn của ảnh.
        - `bbox_inc` : float - tăng kích thước bbox của ảnh
    return:
        - `image`: Ảnh đã crop.
    """

    # image_path = LOCAL_ROOT + image_path

    img = cv2.imread(image_path)  # Đọc ảnh bằng cv2
    # Lấy kích thước của ảnh đầu vào
    real_h, real_w = img.shape[:2]
    assert os.path.exists(
        image_path[:-4] + "_BB.txt"
    ), "path not exists" + " " + image_path  # Kiểm tra xem tệp bbox có tồn tại không

    with open(image_path[:-4] + "_BB.txt", "r") as f:
        material = f.readline()  # Đọc dòng đầu tiên từ tệp bbox
        try:
            x, y, w, h = material.strip().split(" ")[:-1]  # Phân tích cú pháp bbox
        except:
            print(
                "Bounding Box of " + image_path + " is wrong"
            )  # In thông báo lỗi nếu bbox không hợp lệ

        try:
            w = int(
                float(w) * (real_w / 224)
            )  # Chuyển đổi kích thước bbox
            h = int(float(h) * (real_h / 224))
            x = int(float(x) * (real_w / 224))
            y = int(float(y) * (real_h / 224))

            # Crop khuôn mặt dựa trên bounding box của nó
            l = max(w, h)
            # cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), int(l/50))
            xc, yc = x + w / 2, y + h / 2
            x, y = int(xc - l * bbox_inc / 2), int(
                yc - l * bbox_inc / 2
            )  # Tính toán tọa độ mới

            x1 = 0 if x < 0 else x
            y1 = 0 if y < 0 else y
            x2 = (
                real_w if x + l * bbox_inc > real_w else x + int(l * bbox_inc)
            )  # Đảm bảo không vượt quá kích thước ảnh
            y2 = (
                real_h if y + l * bbox_inc > real_h else y + int(l * bbox_inc)
            )
            img = img[y1:y2, x1:x2, :]  # Crop ảnh
            img = cv2.copyMakeBorder(
                img,
                y1 - y,
                int(l * bbox_inc - y2 + y),
                x1 - x,
                int(l * bbox_inc) - x2 + x,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
            )  # Thêm viền đen
        except:
            print(
                "Cropping Bounding Box of " + image_path + " goes wrong"
            )  # In thông báo lỗi nếu crop không thành công

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img  # Trả về ảnh đã crop


def save_image(image_path, img, largest_size, scaleup=True):
    """
    Lưu một ảnh vào image_path

    params:
        - `img`: ảnh cv
        - `image_path` : str - đường dẫn của ảnh để lưu.
        - `largest_size` : int - kích thước của cạnh lớn nhất của hình dạng để lưu
    """
    # Lấy kích thước của ảnh đầu vào
    h, w = img.shape[:2]

    ratio = largest_size / max(h, w)
    if not scaleup:
        ratio = min(ratio, 1.0)

    if ratio != 1.0:
        new_shape = int(w * ratio + 0.01), int(h * ratio + 0.01)
        img = cv2.resize(img, new_shape)  # Thay đổi kích thước ảnh

    return cv2.imwrite(image_path, img)  # Lưu ảnh


def read_orig_labels(orig_dir, spoof_filter=None):
    # đọc nhãn
    org_lbl_dir = orig_dir + "metas/intra_test/"

    train_label = pd.read_json(
        org_lbl_dir + "train_label.json", orient="index"
    ).apply(
        pd.to_numeric, downcast="integer"
    )  # Đọc nhãn huấn luyện từ tệp JSON
    test_label = pd.read_json(
        org_lbl_dir + "test_label.json", orient="index"
    ).apply(
        pd.to_numeric, downcast="integer"
    )  # Đọc nhãn kiểm tra từ tệp JSON

    print("Train / Test shape")
    print(
        "          original: {} / {}".format(
            train_label.shape, test_label.shape
        )
    )  # In kích thước của tập huấn luyện và tập kiểm tra
    # lọc tập dữ liệu với các loại giả mạo được chỉ định
    if spoof_filter:
        train_label = train_label[
            train_label[40].isin(spoof_filter)
        ]  # Lọc nhãn huấn luyện
        test_label = test_label[
            test_label[40].isin(spoof_filter)
        ]  # Lọc nhãn kiểm tra
        print(
            "          filtered: {} / {}".format(
                train_label.shape, test_label.shape
            )
        )  # In kích thước của tập huấn luyện và tập kiểm tra sau khi lọc

    return train_label, test_label  # Trả về nhãn huấn luyện và nhãn kiểm tra


def save_labels(train_label, test_label, dir):
    # data_dir = dir+'data'+str(size)
    train_label.index = train_label.index.str.replace(
        "Data/", ""
    )  # Xóa tiền tố "Data/" khỏi chỉ mục
    test_label.index = test_label.index.str.replace(
        "Data/", ""
    )  # Xóa tiền tố "Data/" khỏi chỉ mục
    pd.concat([train_label, test_label]).to_csv(
        dir + "/label.csv"
    )  # Lưu nhãn vào tệp CSV

    if not os.path.exists(dir + "/train"):
        os.makedirs(dir + "/train")  # Tạo thư mục train nếu nó không tồn tại
    pd.DataFrame(
        {
            "path": train_label.index.str.replace("train/", ""),
            "spoof_type": train_label[40].values,
        }
    ).to_csv(
        dir + "/train/train_target.csv", index=False
    )  # Lưu thông tin về tập huấn luyện vào tệp CSV

    if not os.path.exists(dir + "/test"):
        os.makedirs(dir + "/test")  # Tạo thư mục test nếu nó không tồn tại
    pd.DataFrame(
        {
            "path": test_label.index.str.replace("test/", ""),
            "spoof_type": test_label[40].values,
        }
    ).to_csv(
        dir + "/test/test_target.csv", index=False
    )  # Lưu thông tin về tập kiểm tra vào tệp CSV


def process_images(
    orig_dir, crop_dir, labels, size, bbox_inc=1.5, scaleup=False
):
    for img_path in tqdm(labels):  # Lặp qua tất cả các đường dẫn ảnh
        img = read_image(
            orig_dir + img_path, bbox_inc=bbox_inc
        )  # Đọc và crop ảnh

        new_img_path = img_path.replace(
            "Data", crop_dir
        )  # Tạo đường dẫn mới cho ảnh đã crop
        new_img_dir = os.path.dirname(
            new_img_path
        )  # Lấy thư mục của đường dẫn mới

        if not os.path.exists(new_img_dir):
            os.makedirs(
                new_img_dir
            )  # Tạo thư mục nếu nó không tồn tại
        save_image(
            new_img_path, img, size, scaleup=scaleup
        )  # Lưu ảnh đã crop


if __name__ == "__main__":
    # Phân tích cú pháp các đối số
    def check_zero(value):
        fvalue = float(value)
        if fvalue < 0:
            raise argparse.ArgumentTypeError(
                "%s là một giá trị không hợp lệ" % value
            )
        return fvalue

    p = argparse.ArgumentParser(
        description="Cropping images by bbox"
    )  # Tạo một trình phân tích cú pháp đối số
    p.add_argument(
        "--orig_dir",
        type=str,
        default=CELEBA_DIR,
        help="Directory with original Celeba_Spoof dataset",
    )  # Thêm đối số cho thư mục chứa dữ liệu gốc
    p.add_argument(
        "--crop_dir",
        type=str,
        default=CROP_DIR,
        help="Directory to save cropped dataset",
    )  # Thêm đối số cho thư mục để lưu dữ liệu đã crop
    p.add_argument(
        "--size",
        type=int,
        default=128,
        help="Size of the largest side of the image, px",
    )  # Thêm đối số cho kích thước ảnh
    p.add_argument(
        "--bbox_inc",
        type=check_zero,
        default=1.5,
        help="Image bbox increasing, value 1 makes no effect",
    )  # Thêm đối số cho việc tăng kích thước bbox
    p.add_argument(
        "--spoof_types",
        type=int,
        nargs="+",
        default=list(range(10)),
        help="Spoof types to keep",
    )  # Thêm đối số cho các loại giả mạo để giữ lại
    args = p.parse_args()  # Phân tích cú pháp các đối số dòng lệnh
    if args.orig_dir[-1] != "/":
        args.orig_dir += "/"  # Đảm bảo rằng đường dẫn kết thúc bằng dấu /
    if args.crop_dir[-1] != "/":
        args.crop_dir += "/"  # Đảm bảo rằng đường dẫn kết thúc bằng dấu /

    data_dir = "{}data_{}_{}".format(
        args.crop_dir, args.bbox_inc, args.size
    )  # Tạo đường dẫn cho thư mục dữ liệu
    print("Kiểm tra các đối số:")
    print(
        "    Thư mục dữ liệu gốc       :", args.orig_dir
    )  # In đường dẫn đến thư mục dữ liệu gốc
    print(
        "    Thư mục để lưu ảnh đã crop :", data_dir
    )  # In đường dẫn đến thư mục để lưu ảnh đã crop
    print(
        "    Các loại giả mạo để giữ trong tập dữ liệu   :", args.spoof_types
    )  # In các loại giả mạo để giữ lại
    print(
        "    Kích thước crop, tăng bbox       :", (args.size, args.bbox_inc)
    )  # In kích thước crop và việc tăng bbox

    # Xử lý ảnh
    proceed = (
        input("\nTiến hành? [y/n] : ").lower()[:1] == "y"
    )  # Hỏi người dùng có muốn tiếp tục hay không
    if proceed:
        # Đọc và lọc nhãn
        print("\nĐọc và lọc nhãn...")
        train_label, test_label = read_orig_labels(
            args.orig_dir, spoof_filter=args.spoof_types
        )  # Đọc và lọc nhãn

        # Đọc, Crop, Lưu ảnh
        print("\nĐang xử lý dữ liệu huấn luyện...")
        process_images(
            args.orig_dir,
            data_dir,
            train_label.index,
            args.size,
            bbox_inc=args.bbox_inc,
        )  # Xử lý ảnh huấn luyện
        print("\nĐang xử lý dữ liệu kiểm tra...")
        process_images(
            args.orig_dir,
            data_dir,
            test_label.index,
            args.size,
            bbox_inc=args.bbox_inc,
        )  # Xử lý ảnh kiểm tra

        # Viết nhãn
        print("\nĐang viết nhãn...")
        save_labels(
            train_label, test_label, data_dir
        )  # Lưu nhãn

        print("\nHoàn thành\n")  # In thông báo hoàn thành

    else:
        print("\nĐã hủy\n")  # In thông báo đã hủy


# def read_image(image_path, bbox_inc = 0.3):
#     """
#     Read an image from input path and crop it with bbox

#     params:
#         - `image_path` : str - the path of image.
#         - `bbox_inc` : float - image bbox increasing
#     return:
#         - `image`: Cropped image.
#     """

#     #image_path = LOCAL_ROOT + image_path

#     img = cv2.imread(image_path)
#     # Get the shape of input image
#     real_h, real_w = img.shape[:2]
#     assert os.path.exists(image_path[:-4] + '_BB.txt'), 'path not exists' + ' ' + image_path

#     with open(image_path[:-4] + '_BB.txt','r') as f:
#         material = f.readline()
#         try:
#             x, y, w, h = material.strip().split(' ')[:-1]
#         except:
#             logging.info('Bounding Box of' + ' ' + image_path + ' ' + 'is wrong')

#         try:
#             w = int( float(w)*(real_w / 224) )
#             h = int( float(h)*(real_h / 224) )
#             x = int( float(x)*(real_w / 224) - bbox_inc/2*w )
#             y = int( float(y)*(real_h / 224) - bbox_inc/2*h )
#             # Crop face based on its bounding box
#             x1 = 0 if x < 0 else x
#             y1 = 0 if y < 0 else y
#             x2 = real_w if x1 + (1+bbox_inc)*w > real_w else int(x + (1+bbox_inc)*w)
#             y2 = real_h if y1 + (1+bbox_inc)*h > real_h else int(y + (1+bbox_inc)*h)
#             img = img[y1:y2,x1:x2,:]

#         except:
#             logging.info('Cropping Bounding Box of' + ' ' + image_path + ' ' + 'goes wrong')

#     #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     return img
