import os
import logging
import random
import sys
import time
import numpy as np
import paddle
import paddle.nn as nn
from paddle.vision import transforms as T

from dataset import JointCtrl
from model import build_model

class AverageMeter():
    """ Meter for monitoring losses"""
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.reset()

    def reset(self):
        """reset all values to zeros"""
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        """update avg by val and n, where val is the avg of n values"""
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def get_logger(filename, logger_name=None):
    """set logging file and format
    Args:
        filename: str, full path of the logger file to write
        logger_name: str, the logger name, e.g., 'master_logger', 'local_logger'
    Return:
        logger: python logger
    """
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt="%m%d %I:%M:%S %p")
    # different name is needed when creating multiple logger in one process
    logger = logging.getLogger(logger_name)
    fh = logging.FileHandler(os.path.join(filename))
    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)
    return logger


def train(dataloader,
          model,
          criterion,
          optimizer,
          epoch,
          total_epochs,
          total_batch,
          debug_steps=100,
        #   accum_iter=1,
        #   mixup_fn=None,
        #   amp=False,
          logger=None):
    """Training for one epoch
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: nn.Layer, a ViT model
        criterion: nn.criterion
        epoch: int, current epoch
        total_epochs: int, total num of epochs
        total_batch: int, total num of batches for one epoch
        debug_steps: int, num of iters to log info, default: 100
        accum_iter: int, num of iters for accumulating gradients, default: 1
        mixup_fn: Mixup, mixup instance, default: None
        amp: bool, if True, use mix precision training, default: False
        logger: logger for logging, default: None
    Returns:
        train_loss_meter.avg: float, average loss on current process/gpu
        train_acc_meter.avg: float, average top1 accuracy on current process/gpu
        train_time: float, training time
    """
    model.train()
    train_loss_meter = AverageMeter()
    # train_acc_meter = AverageMeter()

    # if amp is True:
    #     scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
    time_st = time.time()


    for batch_id, data in enumerate(dataloader):
        joint = data[0]
        ctrl = data[1]
        ctrl_orig = ctrl.clone()

        # if mixup_fn is not None:
        #     joint, ctrl = mixup_fn(joint, ctrl_orig)

        # if amp is True: # mixed precision training
        #     with paddle.amp.auto_cast():
        #         output = model(joint)
        #         loss = criterion(output, ctrl)
        #     scaled = scaler.scale(loss)
        #     scaled.backward()

        #     if ((batch_id + 1) % accum_iter == 0) or (batch_id + 1 == len(dataloader)):
        #         scaler.minimize(optimizer, scaled)
        #         optimizer.clear_grad()

        # else:
        output = model(joint)
        loss = criterion(output, ctrl)
        # NOTE: division may be needed depending on the loss function
        # Here no division is needed:
        # default 'reduction' param in nn.CrossEntropyLoss is set to 'mean'
        # loss =  loss / accum_iter
        loss.backward()

        # if ((batch_id +1) % accum_iter == 0) or (batch_id + 1 == len(dataloader)):
        optimizer.step()
        optimizer.clear_grad()

        # pred = F.softmax(output)
        # if mixup_fn:
        #     acc = paddle.metric.accuracy(pred, ctrl_orig)
        # else:
        #     acc = paddle.metric.accuracy(pred, ctrl_orig.unsqueeze(1))

        batch_size = joint.shape[0]
        train_loss_meter.update(loss.numpy()[0], batch_size)
        # train_acc_meter.update(acc.numpy()[0], batch_size)

        if logger and batch_id % debug_steps == 0:
            logger.info(
                f"Epoch[{epoch:03d}/{total_epochs:03d}], " +
                f"Step[{batch_id:04d}/{total_batch:04d}], " +
                f"Avg Loss: {train_loss_meter.avg:e}")
                # f"Avg Acc: {train_acc_meter.avg:.4f}")

    train_time = time.time() - time_st
    return train_loss_meter.avg, train_time

def validate(dataloader, model, criterion, total_batch, debug_steps=100, logger=None):
    """Validation for whole dataset
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: nn.Layer, a ViT model
        criterion: nn.criterion
        total_batch: int, total num of batches for one epoch
        debug_steps: int, num of iters to log info, default: 100
        logger: logger for logging, default: None
    Returns:
        val_loss_meter.avg: float, average loss on current process/gpu
        val_acc1_meter.avg: float, average top1 accuracy on current process/gpu
        val_acc5_meter.avg: float, average top5 accuracy on current process/gpu
        val_time: float, valitaion time
    """
    model.eval()
    val_loss_meter = AverageMeter()
    # val_acc1_meter = AverageMeter()
    # val_acc5_meter = AverageMeter()
    time_st = time.time()

    with paddle.no_grad():
        for batch_id, data in enumerate(dataloader):
            joint = data[0]
            ctrl = data[1]

            output = model(joint)
            loss = criterion(output, ctrl)

            # pred = F.softmax(output)
            # acc1 = paddle.metric.accuracy(pred, ctrl.unsqueeze(1))
            # acc5 = paddle.metric.accuracy(pred, ctrl.unsqueeze(1), k=5)

            batch_size = joint.shape[0]
            val_loss_meter.update(loss.numpy()[0], batch_size)
            # val_acc1_meter.update(acc1.numpy()[0], batch_size)
            # val_acc5_meter.update(acc5.numpy()[0], batch_size)

            if logger and batch_id % debug_steps == 0:
                logger.info(
                    f"Val Step[{batch_id:04d}/{total_batch:04d}], " +
                    f"Avg Loss: {val_loss_meter.avg:e}" )


    val_time = time.time() - time_st
    return val_loss_meter.avg,  val_time

def get_train_transforms():
    return T.Compose([
        # T.ToTensor(),
        T.Normalize(mean=[0.00174034, -0.0296329, 0.08699034, 0.4884157, 0.13937208, 0.1928098], 
                    std=[0.47395173, 0.5340112, 0.62341311, 17.59106008, 7.27811802, 9.06478785])])

def main():
    # 0. Preparation
    # config is updated by: (1) config.py, (2) yaml file, (3) arguments
    # set output folder
    # data_root_path = r'D:\Project\Datasets\randomSampling_bone_controller\npy2'
    data_root_path = r'D:\Project\Datasets\randomSampling'
    output_path = './output_'
    value_0 = np.load("./value_0.npy")

    # mean = np.array([0.00174034, -0.0296329, 0.08699034, 
    #                 0.4884157, 0.13937208, 0.1928098])
    # std = np.array([0.47395173, 0.5340112, 0.62341311, 
    #                 17.59106008, 7.27811802, 9.06478785])

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    seed = 123
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    logger = get_logger(filename=os.path.join(output_path, 'log.txt'))

    NUM_EPOCHS = 1000
    REPORT_FREQ = 20
    SAVE_FREQ = 100
    VALIDATE_FREQ = 20
    batch_size = 300
    last_epoch = 0
    resume_path = None
    if last_epoch > 0:
        resume_path = os.path.join(output_path, f"epoch_{last_epoch}")
    
    lr = 0.001


    joint_path_list = []
    ctrl_path_list = []

    data_len = 100
    for i in range(data_len):
        joint_path_list.append(os.path.join(data_root_path, f"dataBoneSet{i}.npy"))
        ctrl_path_list.append(os.path.join(data_root_path, f"dataCtrlSet_{i}.npy"))

    transform = get_train_transforms()
    dataset_train = JointCtrl(joint_path_list[:int(0.8*data_len)], 
                                ctrl_path_list[:int(0.8*data_len)], 
                                value_0=value_0, 
                                transform=transform)
    dataset_val = JointCtrl(joint_path_list[int(0.8*data_len):], 
                                ctrl_path_list[int(0.8*data_len):], 
                                value_0=value_0, 
                                transform=transform)

    train_dataloader = paddle.io.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader   = paddle.io.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = build_model(model_name='resnet50', in_features=6, out_features=220)
    
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=lr, T_max=NUM_EPOCHS, verbose=False)
    optimizer = paddle.optimizer.AdamW(learning_rate=scheduler, parameters=model.parameters())
    criterion = paddle.nn.MSELoss()

    if resume_path:
        assert os.path.isfile(resume_path + '/model.pdparams') is True
        assert os.path.isfile(resume_path + '/optimize.pdopt') is True
        model_state = paddle.load(resume_path + '/model.pdparams')
        model.set_dict(model_state)
        opt_state = paddle.load(resume_path + '/optimize.pdopt')
        optimizer.set_state_dict(opt_state)
        logger.info(
            f"----- Resume: Load model and optmizer from {resume_path}")

    for epoch in range(last_epoch+1, NUM_EPOCHS+1):
        # train
        # logger.info(f"Now training epoch {epoch}. LR={optimizer.get_lr():.6f}")
        train_loss, train_time = train(dataloader=train_dataloader,
                                                model=model,
                                                criterion=criterion,
                                                optimizer=optimizer,
                                                epoch=epoch,
                                                total_epochs=NUM_EPOCHS,
                                                total_batch=len(train_dataloader),
                                                debug_steps=REPORT_FREQ,
                                                # accum_iter=config.TRAIN.ACCUM_ITER,
                                                # mixup_fn=mixup_fn,
                                                # amp=config.AMP,
                                                logger=logger)
        scheduler.step()

        logger.info(f"----- Epoch[{epoch:03d}/{NUM_EPOCHS:03d}], " +
                    f"Train Loss: {train_loss:e}, " +
                    # f"Train Acc: {train_acc:.4f}, " +
                    f"LR: {optimizer.get_lr():e}, " +
                    f"time: {train_time:.2f}")
        
        if epoch % VALIDATE_FREQ == 0 or epoch == NUM_EPOCHS:
            logger.info(f'----- Validation after Epoch: {epoch}')
            val_loss, val_time = validate(
                dataloader=val_dataloader,
                model=model,
                criterion=criterion,
                total_batch=len(val_dataloader),
                debug_steps=REPORT_FREQ,
                logger=logger)
            logger.info(f"----- Epoch[{epoch:03d}/{NUM_EPOCHS:03d}], " +
                        f"Validation Loss: {val_loss:e}, " +
                        # f"Validation Acc@1: {val_acc1:.4f}, " +
                        # f"Validation Acc@5: {val_acc5:.4f}, " +
                        f"time: {val_time:.2f}")


        if epoch % SAVE_FREQ == 0:
            paddle.save(model.state_dict(), f"{output_path}/epoch_{epoch}/model.pdparams")
            paddle.save(optimizer.state_dict(), f"{output_path}/epoch_{epoch}/optimize.pdopt")
            logger.info(f"----- Save: Saved model and optmizer to {output_path}/epoch_{epoch}")

if __name__ == '__main__':
    main()

# import paddle
# from build_model import ResNet, MiniModel
# from dataset import JointCtrl

# # joint_train_path = "./data/jointAnimation_train.txt"
# # ctrl_train_path = "./data/ctrlAnimation_train.txt"
# # data_train = JointCtrl(joint_train_path, ctrl_train_path)
# # joint_test_path = "./data/jointAnimation_test.txt"
# # ctrl_test_path = "./data/ctrlAnimation_test.txt"
# # data_test = JointCtrl(joint_test_path, ctrl_test_path)
# # joint_predict_path = "./data/jointAnimation_pred.txt"
# # ctrl_predict_path = "./data/ctrlAnimation_pred.txt"
# # data_predict = JointCtrl(joint_predict_path, ctrl_predict_path)

# joint_train_path = "./randomSampling/dataBoneSet0.npy"
# ctrl_train_path = "./randomSampling/dataCtrlSet_0.npy"
# data_train = JointCtrl(joint_train_path, ctrl_train_path)
# joint_test_path = "./randomSampling/dataBoneSet1.npy"
# ctrl_test_path = "./randomSampling/dataCtrlSet_1.npy"
# data_test = JointCtrl(joint_test_path, ctrl_test_path)
# joint_predict_path = "./randomSampling/dataBoneSet2.npy"
# ctrl_predict_path = "./randomSampling/dataCtrlSet_2.npy"
# data_predict = JointCtrl(joint_predict_path, ctrl_predict_path)

# # net=ResNet(depth=152)
# # paddle.summary(net, input_size=(1, 1, 887, 6))
# net = MiniModel(in_features=5322, out_features=220)


# model = paddle.Model(net)
# # schedule = paddle.optimizer.lr.PiecewiseDecay(boundaries=[10, 100, 200, 500], values=[0.01, 0.001, 0.0001, 1e-5, 1e-6], verbose=False)
# schedule = paddle.optimizer.lr.ExponentialDecay(learning_rate=0.0001, gamma=0.9, verbose=False)
# model.prepare(paddle.optimizer.Adam(learning_rate=schedule, parameters=model.parameters()), 
#               paddle.nn.MSELoss(),)
#             #   paddle.metric.Accuracy())


# model.fit(train_data=data_train, eval_data=data_test, save_dir="./output",save_freq=1000,
#             epochs=5000, eval_freq=1, batch_size=16, verbose=1,shuffle=False)
