import numpy as np
import cv2
import time
import math


def xyxy2xywh(x: np.array):
    # Chuyển đổi hộp bounding nx4 từ [[x1, y1, x2, y2]] sang [[x, y, w, h]] 
    # với xy1 = góc trên bên trái, xy2 = góc dưới bên phải
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # tâm x
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # tâm y
    y[:, 2] = x[:, 2] - x[:, 0]  # chiều rộng
    y[:, 3] = x[:, 3] - x[:, 1]  # chiều cao
    return y


def xywh2xyxy(x: np.array):
    # Chuyển đổi hộp bounding nx4 từ [[x, y, w, h]] sang [[x1, y1, x2, y2]] 
    # với xy1 = góc trên bên trái, xy2 = góc dưới bên phải
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # góc trên bên trái x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # góc trên bên trái y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # góc dưới bên phải x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # góc dưới bên phải y
    return y


def xywhn2xyxy(x: np.array, w=640, h=640, padw=0, padh=0):
    # Chuyển đổi hộp bounding nx4 từ [[x, y, w, h]] được chuẩn hóa sang [[x1, y1, x2, y2]] 
    # với xy1 = góc trên bên trái, xy2 = góc dưới bên phải
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # góc trên bên trái x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # góc trên bên trái y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # góc dưới bên phải x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # góc dưới bên phải y
    return y


def xyn2xy(x: np.array, w=640, h=640, padw=0, padh=0):
    # Chuyển đổi các đoạn chuẩn hóa thành các đoạn pixel, kích thước (n,2)
    y = np.copy(x)
    y[:, 0] = w * x[:, 0] + padw  # góc trên bên trái x
    y[:, 1] = h * x[:, 1] + padh  # góc trên bên trái y
    return y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Chuyển đổi tỉ lệ tọa độ (xyxy) từ img1_shape sang img0_shape
    if ratio_pad is None:  # tính toán từ img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain = cũ / mới
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # bù chiều rộng và chiều cao
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # bù x
    coords[:, [1, 3]] -= pad[1]  # bù y
    coords[:, :4] /= gain  # chia theo gain
    clip_coords(coords, img0_shape)  # cắt các tọa độ để không vượt ra ngoài kích thước ảnh
    return coords


def clip_coords(boxes, img_shape):
    # Cắt các hộp bounding xyxy để chúng nằm trong kích thước ảnh (chiều cao, chiều rộng)
    boxes[:, 0] = np.clip(0, boxes[:, 0], img_shape[1])  # giới hạn x1
    boxes[:, 1] = np.clip(0, boxes[:, 1], img_shape[0])  # giới hạn y1
    boxes[:, 2] = np.clip(0, boxes[:, 2], img_shape[1])  # giới hạn x2
    boxes[:, 3] = np.clip(0, boxes[:, 3], img_shape[0])  # giới hạn y2


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Tính toán chỉ số giao nhau trên hợp nhất (IoU) giữa hai hộp.
    Cả hai tập hợp hộp đều có dạng (x1, y1, x2, y2).
    """
    def box_area(box):
        # tính diện tích của một hộp
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = np.minimum(box1[:, None, 2:], box2[:, 2:]) - np.maximum(box1[:, None, :2], box2[:, :2])
    inter = np.clip(0, inter, np.max(inter))
    inter = inter.prod(2)
    return inter / (area1[:, None] + area2 - inter)  # IoU = giao / (diện tích1 + diện tích2 - giao)


def wh_iou(wh1, wh2):
    # Trả về ma trận IoU nxm. wh1 là nx2, wh2 là mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    # inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    inter = np.prod(np.minimum(wh1, wh2), 2)
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # IoU = giao / (diện tích1 + diện tích2 - giao)


def nms(boxes, scores, threshold):
    assert boxes.shape[0] == scores.shape[0]
    # gốc dưới bên trái
    ys1 = boxes[:, 0]
    xs1 = boxes[:, 1]
    # góc trên bên phải
    ys2 = boxes[:, 2]
    xs2 = boxes[:, 3]
    # phạm vi tọa độ của hộp bao gồm cả hai đầu
    areas = (ys2 - ys1) * (xs2 - xs1)
    scores_indexes = scores.argsort().tolist()
    boxes_keep_index = []
    while len(scores_indexes):
        index = scores_indexes.pop()
        boxes_keep_index.append(index)
        if not len(scores_indexes):
            break
        ious = compute_iou(boxes[index], boxes[scores_indexes], areas[index],
                           areas[scores_indexes])
        filtered_indexes = set((ious > threshold).nonzero()[0])
        # nếu không còn chỉ số scores_index
        # thì chúng ta sẽ loại bỏ nó
        scores_indexes = [
            v for (i, v) in enumerate(scores_indexes)
            if i not in filtered_indexes
        ]
    return np.array(boxes_keep_index)


def compute_iou(box, boxes, box_area, boxes_area):
    assert boxes.shape[0] == boxes_area.shape[0]
    ys1 = np.maximum(box[0], boxes[:, 0])
    xs1 = np.maximum(box[1], boxes[:, 1])
    ys2 = np.minimum(box[2], boxes[:, 2])
    xs2 = np.minimum(box[3], boxes[:, 3])
    intersections = np.maximum(ys2 - ys1, 0) * np.maximum(xs2 - xs1, 0)
    unions = box_area + boxes_area - intersections
    ious = intersections / unions
    return ious


def non_max_suppression(prediction, conf_thresh=0.25, iou_thresh=0.45, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Thực hiện Nén cực đại (NMS) trên kết quả suy luận

    Trả về:
         danh sách các phát hiện, mỗi hình ảnh có tensor (n,6) [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # số lượng lớp
    xc = prediction[..., 4] > conf_thresh  # ứng viên

    # Kiểm tra
    assert 0 <= conf_thresh <= 1, f'Invalid Confidence threshold {conf_thresh}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thresh <= 1, f'Invalid IoU {iou_thresh}, valid values are between 0.0 and 1.0'

    # Cài đặt
    min_wh, max_wh = 2, 4096  # (pixel) chiều rộng và chiều cao hộp tối thiểu và tối đa
    max_nms = 30000  # số hộp tối đa vào torchvision.ops.nms()
    time_limit = 10.0  # giới hạn thời gian là 10 giây
    redundant = True  # yêu cầu phát hiện dư thừa
    multi_label &= nc > 1  # nhiều nhãn cho mỗi hộp (thêm 0.5ms/img)
    merge = False  # sử dụng merge-NMS

    t = time.time()
    output = [np.zeros((0, 6))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # chỉ số hình ảnh, suy luận trên hình ảnh
        # Áp dụng các ràng buộc
        x = x[xc[xi]]  # confidence

        # Gắn nhãn trước nếu tự động gắn nhãn
        if labels and len(labels[xi]):
            l = labels[xi]
            v = np.zeros((len(l), nc + 5))
            v[:, :4] = l[:, 1:5]  # hộp
            v[:, 4] = 1.0  # độ tin cậy
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # lớp
            x = np.concatenate((x, v), axis=0)

        # Nếu không còn gì, tiếp tục xử lý ảnh tiếp theo
        if not x.shape[0]:
            continue

        # Tính toán độ tin cậy
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Hộp (center x, center y, width, height) sang (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        # Ma trận phát hiện nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thresh).nonzero(as_tuple=False).T
            x = np.concatenate((box[i], x[i, j + 5, None], j[:, None]), axis=1)
        else:  # chỉ lớp tốt nhất
            conf = x[:, 5:].max(1, keepdims=True)
            j = x[:, 5:].argmax(1)[..., None]
            x = np.concatenate((box, conf, j), axis=1)[conf[:, 0] > conf_thresh]

        # Kiểm tra kích thước
        n = x.shape[0]  # số lượng hộp
        if not n:  # không có hộp
            continue
        elif n > max_nms:  # quá nhiều hộp
            x = x[x[:, 4].argsort()][::-1][:max_nms]  # sắp xếp theo độ tin cậy

        # NMS theo lô
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # các lớp
        boxes, scores = x[:, :4] + c, x[:, 4]  # hộp (bù theo lớp), điểm số
        i = nms(boxes, scores, iou_thresh)
        if i.shape[0] > max_det:  # giới hạn phát hiện
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (hộp được gộp sử dụng trung bình có trọng số)
            iou = box_iou(boxes[i], boxes) > iou_thresh  # ma trận IoU
            weights = iou * scores[None]  # trọng số hộp
            x[i, :4] = np.matmul(weights, x[:, :4]) / np.sum(weights, axis=1, keepdims=True) # hộp đã gộp
            if redundant:
                i = i[iou.sum(1) > 1]  # yêu cầu dư thừa

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'CẢNH BÁO: Quá thời gian NMS {time_limit}s')
            break  # vượt quá thời gian
    return output


def letterbox(img, new_shape=(416, 416), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    # Thay đổi kích thước và thêm padding vào hình ảnh trong khi đáp ứng các ràng buộc về bội số stride
    shape = img.shape[:2]  # kích thước hiện tại [chiều cao, chiều rộng]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Tỉ lệ thay đổi (mới / cũ)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # chỉ thay đổi kích thước xuống, không thay đổi kích thước lên (để kiểm tra mAP tốt hơn)
        r = min(r, 1.0)

    # Tính toán padding
    ratio = r, r  # tỉ lệ chiều rộng và chiều cao
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # padding chiều rộng và chiều cao
    if auto:  # hình chữ nhật nhỏ nhất
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # padding chiều rộng và chiều cao
    elif scaleFill:  # kéo dài
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # tỉ lệ chiều rộng và chiều cao

    dw /= 2  # chia padding vào 2 bên
    dh /= 2

    if shape[::-1] != new_unpad:  # thay đổi kích thước
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)  # padding trên dưới
    left, right = round(dw - 0.1), round(dw + 0.1)  # padding trái phải
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # border
    return img, ratio, (dw, dh)