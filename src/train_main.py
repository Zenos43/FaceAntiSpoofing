import torch
from torch import optim
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm
from tensorboardX import SummaryWriter

from src.NN import MultiFTNet
from src.dataset_loader import get_train_valid_loader
from datetime import datetime


class TrainMain:
    def __init__(self, conf):
        """
        Khởi tạo lớp TrainMain với các tham số cấu hình.

        Args:
            conf: Đối tượng chứa các tham số cấu hình.
        """
        self.conf = conf
        self.step = 0  # Khởi tạo bước huấn luyện
        self.val_step = 0  # Khởi tạo bước đánh giá
        self.start_epoch = 0  # Khởi tạo epoch bắt đầu
        self.train_loader, self.valid_loader = get_train_valid_loader(
            self.conf
        )  # Lấy train và valid dataloader
        self.board_train_every = (
            len(self.train_loader) // conf.board_loss_per_epoch
        )  # Tần suất ghi log train
        self.board_valid_every = (
            len(self.valid_loader) // conf.board_loss_per_epoch
        )  # Tần suất ghi log valid

    def train_model(self):
        """
        Huấn luyện mô hình.
        """
        self._init_model_param()  # Khởi tạo các tham số mô hình
        self._train_stage()  # Bắt đầu giai đoạn huấn luyện

    def _init_model_param(self):
        """
        Khởi tạo các tham số cần thiết cho mô hình.
        """
        self.cls_criterion = CrossEntropyLoss()  # Hàm mất mát cho bài toán phân loại
        self.ft_criterion = MSELoss()  # Hàm mất mát cho bài toán feature matching
        self.model = self._define_network()  # Định nghĩa kiến trúc mạng
        self.optimizer = optim.SGD(
            self.model.module.parameters(),
            lr=self.conf.lr,
            weight_decay=5e-4,
            momentum=self.conf.momentum,
        )  # Khởi tạo optimizer

        self.schedule_lr = optim.lr_scheduler.MultiStepLR(
            self.optimizer, self.conf.milestones, self.conf.gamma, -1
        )  # Khởi tạo scheduler để thay đổi learning rate

        print("lr: ", self.conf.lr)
        print("epochs: ", self.conf.epochs)
        print("milestones: ", self.conf.milestones)

    def _train_stage(self):
        """
        Giai đoạn huấn luyện mô hình.
        """
        run_loss = 0.0  # Khởi tạo loss trung bình
        run_acc = 0.0  # Khởi tạo độ chính xác trung bình
        run_loss_cls = 0.0  # Khởi tạo loss phân loại trung bình
        run_loss_ft = 0.0  # Khởi tạo loss feature matching trung bình
        run_val_acc = 0.0  # Khởi tạo độ chính xác validation trung bình
        run_val_loss_cls = 0.0  # Khởi tạo loss validation trung bình

        is_first = True  # Kiểm tra xem có phải lần đầu huấn luyện không

        print("Board train loss every {} steps".format(self.board_train_every))
        print("Board valid loss every {} steps".format(self.board_valid_every))
        for e in range(self.start_epoch, self.conf.epochs):
            if is_first:
                self.writer = SummaryWriter(
                    self.conf.log_path
                )  # Khởi tạo tensorboard writer
                is_first = False

            # Training
            print(
                "Epoch {} started. lr: {}".format(
                    e, self.schedule_lr.get_last_lr()
                )
            )
            self.model.train()  # Chuyển mô hình sang chế độ train
            print("Training on {} batches.".format(len(self.train_loader)))
            for sample, ft_sample, labels in tqdm(iter(self.train_loader)):
                imgs = [sample, ft_sample]

                loss, acc, loss_cls, loss_ft = self._train_batch_data(
                    imgs, labels
                )  # Huấn luyện trên một batch
                run_loss += loss
                run_acc += acc
                run_loss_cls += loss_cls
                run_loss_ft += loss_ft

                self.step += 1

                if (
                    self.step % self.board_train_every == 0 and self.step != 0
                ):  # Ghi log sau mỗi board_train_every bước
                    board_step = self.step // self.board_train_every
                    self.writer.add_scalar(
                        "Loss/train", run_loss / self.board_train_every, board_step
                    )
                    self.writer.add_scalar(
                        "Acc/train", run_acc / self.board_train_every, board_step
                    )
                    self.writer.add_scalar(
                        "Loss_cls/train",
                        run_loss_cls / self.board_train_every,
                        board_step,
                    )
                    self.writer.add_scalar(
                        "Loss_ft/train",
                        run_loss_ft / self.board_train_every,
                        board_step,
                    )
                    self.writer.add_scalar(
                        "Learning_rate",
                        self.optimizer.param_groups[0]["lr"],
                        board_step,
                    )

                    run_loss = 0.0
                    run_acc = 0.0
                    run_loss_cls = 0.0
                    run_loss_ft = 0.0

            self.schedule_lr.step()  # Cập nhật learning rate

            # Validation
            self.model.eval()  # Chuyển mô hình sang chế độ eval
            print("Validation on {} batches.".format(len(self.valid_loader)))
            for sample, labels in tqdm(iter(self.valid_loader)):
                with torch.no_grad():  # Tắt tính toán gradient
                    acc, loss_cls = self._valid_batch_data(
                        sample, labels
                    )  # Đánh giá trên một batch
                run_val_acc += acc
                run_val_loss_cls += loss_cls

                self.val_step += 1

                if (
                    self.val_step % self.board_valid_every == 0
                    and self.val_step != 0
                ):  # Ghi log sau mỗi board_valid_every bước
                    board_step = self.val_step // self.board_valid_every
                    self.writer.add_scalar(
                        "Acc/valid", run_val_acc / self.board_valid_every, board_step
                    )
                    self.writer.add_scalar(
                        "Loss_cls/valid",
                        run_val_loss_cls / self.board_valid_every,
                        board_step,
                    )
                    run_val_acc = 0.0
                    run_val_loss_cls = 0.0

            self._save_state("epoch-{}".format(e))  # Lưu lại trạng thái mô hình

        self.writer.close()  # Đóng tensorboard writer

    def _train_batch_data(self, imgs, labels):
        """
        Huấn luyện mô hình trên một batch dữ liệu.

        Args:
            imgs: Danh sách chứa ảnh và feature map.
            labels: Nhãn của ảnh.

        Returns:
            loss.item(): Giá trị loss.
            acc: Độ chính xác.
            loss_cls.item(): Giá trị loss phân loại.
            loss_fea.item(): Giá trị loss feature matching.
        """
        self.optimizer.zero_grad()  # Xóa gradient
        labels = labels.to(self.conf.device)  # Chuyển nhãn lên device
        embeddings, feature_map = self.model.forward(
            imgs[0].to(self.conf.device)
        )  # Forward pass

        loss_cls = self.cls_criterion(
            embeddings, labels
        )  # Tính loss phân loại
        loss_fea = self.ft_criterion(
            feature_map, imgs[1].to(self.conf.device)
        )  # Tính loss feature matching

        loss = 0.5 * loss_cls + 0.5 * loss_fea  # Tính loss tổng
        acc = self._get_accuracy(embeddings, labels)[
            0
        ]  # Tính độ chính xác
        loss.backward()  # Backward pass
        self.optimizer.step()  # Cập nhật tham số
        return loss.item(), acc, loss_cls.item(), loss_fea.item()

    def _valid_batch_data(self, img, labels):
        """
        Đánh giá mô hình trên một batch dữ liệu.

        Args:
            img: Ảnh đầu vào.
            labels: Nhãn của ảnh.

        Returns:
            acc: Độ chính xác.
            loss_cls.item(): Giá trị loss phân loại.
        """
        labels = labels.to(self.conf.device)  # Chuyển nhãn lên device
        embeddings = self.model.forward(
            img.to(self.conf.device)
        )  # Forward pass

        loss_cls = self.cls_criterion(
            embeddings, labels
        )  # Tính loss phân loại
        acc = self._get_accuracy(embeddings, labels)[
            0
        ]  # Tính độ chính xác

        return acc, loss_cls.item()

    def _define_network(self):
        """
        Định nghĩa kiến trúc mạng.

        Returns:
            model: Mô hình mạng.
        """
        param = {
            "num_classes": self.conf.num_classes,
            "img_channel": self.conf.input_channel,
            "embedding_size": self.conf.embedding_size,
            "conv6_kernel": self.conf.kernel_size,
        }

        model = MultiFTNet(**param).to(self.conf.device)  # Khởi tạo mô hình
        model = torch.nn.DataParallel(
            model
        )  # self.conf.devices)  # Sử dụng DataParallel để huấn luyện trên nhiều GPU
        model.to(self.conf.device)  # Chuyển mô hình lên device
        return model

    def _get_accuracy(self, output, target, topk=(1,)):
        """
        Tính độ chính xác.

        Args:
            output: Output của mô hình.
            target: Nhãn thật.
            topk: Danh sách các giá trị k để tính top-k accuracy.

        Returns:
            ret: Danh sách độ chính xác top-k.
        """
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        ret = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
            ret.append(correct_k.mul_(1.0 / batch_size))
        return ret

    def _save_state(self, stage):
        """
        Lưu lại trạng thái của mô hình.

        Args:
            stage: Tên của stage (ví dụ: epoch-1).
        """
        save_path = self.conf.model_path
        job_name = self.conf.job_name
        time_stamp = (str(datetime.now())[:-10]).replace(" ", "-").replace(
            ":", "-"
        )
        torch.save(
            self.model.state_dict(),
            save_path
            + "/"
            + ("{}_{}_{}.pth".format(time_stamp, job_name, stage)),
        )
```