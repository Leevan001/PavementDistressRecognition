#%% 导库
import os
from datetime import datetime
from config import *
from Model import DenseNetModel
from Dataset import WovenBagDataset

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, lr_scheduler
import torch

#%% 准备自定义数据集
train_dataset = WovenBagDataset(os.path.join(ROOT_DIR, DATA_DIR, TRAIN_DIR), 'train')
valid_dataset = WovenBagDataset(os.path.join(ROOT_DIR, DATA_DIR, TRAIN_DIR), 'valid')

#%% 加载数据
# TODO 加载device
train_data = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_data = DataLoader(train_dataset, batch_size=BATCH_SIZE)

#%% 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenseNetModel
model.to(device)

#%% 损失函数
loss_fuc = CrossEntropyLoss().to(device)

#%% 优化器
optimizer = Adam(model.parameters(), LR)
scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEP, gamma=0.9)  # 设置学习率下降策略

#%% 日志相关
# log dir
now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
log_dir = os.path.join(ROOT_DIR, "results", time_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

#%% 训练前准备
train_curve = list()
valid_curve = list()
log_interval = 1     # 打印训练信息的间隔
val_interval = 1     # 验证的频率
best_eval_acc = [0]  # 验证集的准确率，用于决定是否保存模型

#%% 训练函数
def train(net, epoch):
    net.train()
    loss_mean = 0.
    correct = 0.
    total = 0.
    for i, data in enumerate(train_data):
        # forward
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)       # [1, 2]
        # backward
        optimizer.zero_grad()
        loss = loss_fuc(outputs, labels)
        loss.backward()

        # update weights
        optimizer.step()

        # 统计分类情况
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().cpu().sum().numpy()

        # 打印训练信息
        loss_mean += loss.item()
        train_curve.append(loss.item())
        if (i+1) % log_interval == 0:
            loss_mean = loss_mean / log_interval
            train_acc = correct / total
            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.4%} lr:{:.6f}".format(
                epoch+1, EPOCH, i+1, len(train_data), loss_mean, train_acc, scheduler.get_lr()[0]))
            loss_mean = 0.
    loss_mean /= len(train_data)
    train_acc = correct / total
    print("|Training|Epoch:[{:0>3}/{:0>3}]|Loss: {:.4f}|Acc:{:.4%}|lr:{:.6f}|"
          .format(epoch + 1, EPOCH, loss_mean, train_acc, scheduler.get_lr()[0]))
    scheduler.step()  # 更新学习率

#%% 验证函数
def evaluate(net, epoch):
    net.eval()
    correct_val = 0.
    total_val = 0.
    loss_val = 0.

    with torch.no_grad():
        for j, data in enumerate(valid_data):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = loss_fuc(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).squeeze().cpu().sum().numpy()

            loss_val += loss.item()

        loss_val_mean = loss_val / len(valid_data)
        eval_acc = correct_val / total_val
        valid_curve.append(loss_val_mean)
        print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.4%}".format(
            epoch+1, EPOCH, j + 1, len(valid_data), loss_val_mean, eval_acc))
        if max(best_eval_acc) < eval_acc:
            best_eval_acc.append(eval_acc)
            torch.save(net.state_dict(), log_dir + "{:0>3}.pth".format(epoch + 1))


#%%
if __name__ == "__main__":
    for epoch in range(0, EPOCH):
        train(model, epoch)
        evaluate(model, epoch)
        # 画图
        train_x = range(len(train_curve))
        train_y = train_curve

        train_iters = len(train_data)
        valid_x = np.arange(1, len(
            valid_curve) + 1) * train_iters * val_interval  # 由于valid中记录的是epochloss，需要对记录点进行转换到iterations
        valid_y = valid_curve

        plt.plot(train_x, train_y, label='Train')
        plt.plot(valid_x, valid_y, label='Valid')

        plt.legend(loc='upper right')
        plt.ylabel('loss value')
        plt.xlabel('Iteration')
        plt.savefig(os.path.join(log_dir, "loss_curve.png"))
        plt.show()