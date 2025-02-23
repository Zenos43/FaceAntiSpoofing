import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize
import numpy as np


def spoof_labels_to_classes(labels_df, classes):
    # Tạo một bản sao của DataFrame labels để tránh sửa đổi DataFrame gốc
    df = labels_df.copy()
    # Nếu số lượng lớp là 2
    if len(classes) == 2:
        # Định nghĩa hàm chuyển đổi giá trị thành lớp tương ứng
        def to_class(v):
            # Trả về lớp đầu tiên nếu giá trị là 0, ngược lại trả về lớp thứ hai
            return classes[0] if v == 0 else classes[1]
    # Nếu số lượng lớp là 3
    elif len(classes) == 3:
        # Định nghĩa hàm chuyển đổi giá trị thành lớp tương ứng
        def to_class(v):
            # Nếu giá trị nằm trong [1, 2, 3], trả về lớp thứ hai
            if v in [1, 2, 3]:
                return classes[1]
            # Nếu giá trị nằm trong [7, 8, 9], trả về lớp thứ ba
            if v in [7, 8, 9]:
                return classes[2]
            # Nếu giá trị là 0, trả về lớp đầu tiên
            if v == 0:
                return classes[0]
            # Nếu không thuộc các trường hợp trên, trả về None
            return None
    # Nếu số lượng lớp khác 2 hoặc 3
    else:
        # In thông báo cho biết nhãn sẽ không được thay đổi
        print('Labels will not be changed')
        # Định nghĩa hàm trả về giá trị ban đầu
        def to_class(v):
            return v
    # Áp dụng hàm chuyển đổi cho cột thứ hai của DataFrame
    df.iloc[:, 1] = df.iloc[:, 1].apply(lambda s: to_class(s))

    # Trả về DataFrame đã được chuyển đổi
    return df


def plot_value_counts(
    series,
    n_values=25,
    fillna='NONE',
    figwidth=12,
    bar_thickness=0.5,
    sort_index=False,
    verbose=False,
    show_percents=False,
):
    # Loại bỏ các giá trị NaN và lấy các giá trị duy nhất từ series
    _ = series.dropna().unique()
    # Nếu verbose là True, in ra thông tin về series và các giá trị duy nhất
    if verbose:
        print(
            '`{}`, {} unique values: \n{}'.format(
                series.name, len(_), sorted(_)
            )
        )

    # Đếm số lần xuất hiện của mỗi giá trị trong series, điền giá trị NaN bằng fillna
    val_counts = series.fillna(fillna).value_counts()
    # Nếu sort_index là True, sắp xếp các giá trị theo index
    if sort_index:
        val_counts = val_counts.sort_index()
    # Lấy n_values giá trị đầu tiên
    bar_values = val_counts.values[:n_values]
    # Lấy n_values nhãn đầu tiên và chuyển đổi thành kiểu chuỗi
    bar_labels = val_counts.index[:n_values].astype('str')
    # Tạo figure với kích thước cho trước
    plt.figure(
        figsize=(figwidth, bar_thickness * min(len(val_counts), n_values))
    )
    # Vẽ biểu đồ cột
    ax = sns.barplot(x=bar_values, y=bar_labels)
    # Đặt tiêu đề cho biểu đồ
    ax.set(
        title='"{}" value counts ({} / {})'.format(
            series.name, len(bar_labels), val_counts.shape[0]
        ),
        # xlim=[0, 1.07*bar_values.max()]
    )
    # Nếu show_percents là True, hiển thị phần trăm của mỗi giá trị
    if show_percents:
        labels = [
            f'{w/val_counts.values.sum()*100:0.1f}%'
            if (w := v.get_width()) > 0
            else ''
            for v in ax.containers[0]
        ]
    # Nếu không, hiển thị số lượng giá trị
    else:
        labels = bar_values
    # Thêm nhãn vào các cột
    plt.bar_label(ax.containers[0], labels=labels, label_type='center')
    # Đặt màu cho cột có nhãn là fillna thành màu đen
    for i in range(len(bar_labels)):
        if bar_labels[i] == fillna:
            ax.patches[i].set_color('black')
    # Hiển thị lưới
    plt.grid()
    # Hiển thị biểu đồ
    plt.show()


def plot_iter_images(iter, size, count):
    # Import thư viện torchvision.transforms
    import torchvision.transforms as T

    # Tính số hàng cần thiết để hiển thị count ảnh
    rows = count // 4
    # Kiểm tra xem iter có chứa cả sample và ft_sample hay không
    if len(iter) > 2:
        # Nếu có, gán giá trị cho sample, ft_sample và target
        sample, ft_sample, target = iter
    else:
        # Nếu không, gán giá trị cho sample và target, ft_sample là None
        sample, target = iter
        ft_sample = None
    # Chuyển đổi target thành mảng numpy
    target = target.numpy()
    # Tạo figure với kích thước cho trước
    fig = plt.figure(figsize=(12, 4 * rows))
    # Lặp qua count ảnh
    for i in range(count):
        # Thêm subplot vào figure
        ax = fig.add_subplot(rows, 4, i + 1)
        # Tắt trục
        ax.axis('off')

        # Hiển thị ảnh sample
        plt.imshow(T.ToPILImage()(sample[i]), extent=(0, size, 0, size))
        # Thêm nhãn target vào ảnh
        plt.text(0, -20, target[i], fontsize=20, color='red')
        # Nếu ft_sample không phải là None, hiển thị ảnh ft_sample
        if ft_sample is not None:
            plt.imshow(
                T.ToPILImage()(ft_sample[i]),
                extent=(3 * size / 4, 5 * size / 4, -size / 4, size / 4),
            )
        # Đặt giới hạn cho trục x và y
        plt.xlim(0, 5 * size / 4)
        plt.ylim(-size / 4, size)
        # Xóa các ticks trên trục y và x
        plt.yticks([])
        plt.xticks([])
        # Điều chỉnh bố cục
        plt.tight_layout()
    # Hiển thị figure
    plt.show()


def roc_curve_plots(y_true, model_proba, title='ROC Curve', figsize=(12, 9)):
    ''' Функция построения ROC кривой для моделей из переданнойго словаря

        Принимает:
    '''
    # Tạo figure với kích thước cho trước
    plt.figure(figsize=figsize)
    # Lặp qua các model và proba
    for name, proba in model_proba.items():
        # Tính toán fpr, tpr và ngưỡng
        fpr, tpr, _ = roc_curve(y_true, proba)
        # Tính toán diện tích dưới đường cong ROC
        roc_auc = auc(fpr, tpr)
        # Vẽ đường cong ROC
        plt.plot(
            fpr, tpr, lw=2, label='AUC = %0.4f (%s)' % (roc_auc, name)
        )
    # Đặt tiêu đề cho biểu đồ
    plt.title(title)
    # Hiển thị chú giải
    plt.legend(loc='lower right')
    # Vẽ đường chéo
    plt.plot([0, 1], [0, 1], 'r--')
    # Hiển thị lưới
    plt.grid()
    # Đặt giới hạn cho trục x và y
    plt.xlim([-0.01, 1]), plt.ylim([0, 1.01])
    # Đặt nhãn cho trục y và x
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    # Hiển thị biểu đồ
    plt.show()


def multiclass_roc_curve_plots(
    y_true,
    proba,
    class_labels=None,
    title='ROC Curve',
    figsize=(12, 9),
):
    # Khởi tạo các dictionary để lưu trữ fpr, tpr và roc_auc
    fpr, tpr, roc_auc = {}, {}, {}

    # Nếu class_labels là None, gán giá trị mặc định
    if class_labels is None:
        class_labels = [0, 1, 2]

    # Chuyển đổi y_true thành dạng one-hot encoding
    y_true = label_binarize(y_true, classes=[0, 1, 2])
    # Lấy số lượng lớp
    n_classes = y_true.shape[1]

    # Tính toán fpr, tpr và roc_auc cho micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Tính toán fpr, tpr và roc_auc cho từng lớp
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Lấy tất cả các giá trị fpr duy nhất
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Nội suy tất cả các đường cong ROC tại các điểm này
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Tính trung bình và tính AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Vẽ tất cả các đường cong ROC
    plt.figure(figsize=figsize)
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        linestyle=":",
        lw=4,
        label="AUC = {:0.4f} (micro-average ROC curve)".format(
            roc_auc["micro"]
        ),
    )
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        ls=":",
        lw=4,
        label="AUC = {:0.4f} (macro-average ROC curve)".format(
            roc_auc["macro"]
        ),
    )
    for i in range(n_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            lw=2,
            label="AUC = {:0.4f} (Class: {})".format(
                roc_auc[i], class_labels[i]
            ),
        )
    # Đặt tiêu đề cho biểu đồ
    plt.title(title)
    # Hiển thị chú giải
    plt.legend(loc='lower right')
    # Vẽ đường chéo
    plt.plot([0, 1], [0, 1], 'r--')
    # Hiển thị lưới
    plt.grid()
    # Đặt giới hạn cho trục x và y
    plt.xlim([-0.01, 1]), plt.ylim([0, 1.01])
    # Đặt nhãn cho trục y và x
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    # Hiển thị biểu đồ
    plt.show()


def confusion_matricies_plots(
    confusion_matricies, class_labels=None, figsize=(12, 4)
):
    # Import thư viện pandas
    import pandas as pd

    # Tạo subplot
    _, ax = plt.subplots(1, len(confusion_matricies), figsize=figsize)
    # Khởi tạo biến đếm
    i = 0
    # Lặp qua các ma trận nhầm lẫn
    for name, matrix in confusion_matricies.items():
        # Vẽ heatmap
        sns.heatmap(
            pd.DataFrame(matrix),
            annot=True,
            annot_kws={"fontsize": 12},
            cmap='Blues',
            vmin=0,
            vmax=1,
            ax=ax[i],
        )
        # Đặt tiêu đề cho subplot
        ax[i].set_title(name, fontsize=14)
        # Đặt nhãn cho trục x và y
        ax[i].set_xlabel('Predictions', fontsize=14)
        ax[i].set_ylabel('True', fontsize=14)
        # Nếu class_labels không phải là None, đặt ticks cho trục x và y
        if class_labels is not None:
            ax[i].set_xticks(
                np.arange(0.5, len(class_labels)), class_labels, fontsize=12
            )
            ax[i].set_yticks(
                np.arange(0.5, len(class_labels)), class_labels, fontsize=12
            )
        # Tăng biến đếm
        i += 1
    # Đặt tiêu đề cho figure
    plt.suptitle('Confusion matricies', fontsize=12, y=1.0)
    # Hiển thị biểu đồ
    plt.show()
```