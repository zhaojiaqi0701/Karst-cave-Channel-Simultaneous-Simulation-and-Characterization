

import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers.commands import train
from tensorflow.keras.optimizers import Adam
# from unet.unet3 import UNet3D
import torch.nn.functional as F
from utils import DataGenerator
import configs as configs
from TransUnetMSAA import VisionTransformer
import time
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay  # 添加导入
import matplotlib.pyplot as plt
import seaborn as sns

seed = 12345
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False
# 定义保存检查点的函数，保存为.pth文件
def save_checkpoint(state, filename="SeismicTUMSAA损失优化.pth"):
    torch.save(state, filename)

# 初始化模型、损失函数和优化器
config = configs.get_r50_b16_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionTransformer(config, img_size=128, num_classes=21843, zero_head=False, vis=False).to(device)

# # # # # 加载预训练模型权重
# # pretrained_path = r"/root/autodl-tmp/modelNoiseTU-MSAA/46Seismic+NoiseNoisecheckpointTUMSAA.70.pth" # 替换为实际的预训练模型路径
# # state_dict = torch.load(pretrained_path)
# # # 检查预训练模型是否包含不必要的键
# # if "state_dict" in state_dict:
# #     state_dict = state_dict["state_dict"]
# # model.load_state_dict(state_dict)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 优化后的损失函数方案
# Dice损失实现

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, prediction, target):
        prediction = torch.sigmoid(prediction)
        intersection = (prediction * target).sum()
        union = prediction.sum() + target.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice
class CaveOptimizedLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, cave_class_idx=1):
        super().__init__()
        self.alpha = alpha  # 溶洞损失权重
        self.beta = beta    # 其他损失权重
        self.cave_class_idx = cave_class_idx  # 溶洞类别索引
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        
    def forward(self, outputs, targets):
        # 主交叉熵损失
        ce_loss = self.ce_loss(outputs, targets)
        
        # 溶洞特异性损失
        cave_mask = (targets == self.cave_class_idx).float()
        cave_dice_loss = self.dice_loss(outputs[:, self.cave_class_idx], cave_mask)
        
        # 组合损失
        total_loss = self.alpha * cave_dice_loss + self.beta * ce_loss
        return total_loss

# 使用优化后的损失函数
criterion = CaveOptimizedLoss(alpha=0.3, beta=0.7, cave_class_idx=2)
# 优化器配置
optimizer = optim.Adam(model.parameters(), lr=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=20, min_lr=1e-8)

# 数据路径
sxpath = "/root/autodl-fs/ChannelKarst/seismic"
kxpath = "/root/autodl-fs/ChannelKarst/label"
n1, n2, n3 = 256, 256, 256
tdata_ids = list(range(1, 121))
vdata_ids = list(range(121, 141))
params = {'dim': (n1, n2, n3), 'n_channels': 1}
train_dataset = DataGenerator(dpath=sxpath, fpath=kxpath, data_IDs=tdata_ids, **params)
valid_dataset = DataGenerator(dpath=sxpath, fpath=kxpath, data_IDs=vdata_ids, **params)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

# TensorBoard日志记录
writer = SummaryWriter(log_dir='./log')
# 训练循环
epochs = 200

def plot_confusion_matrix(y_true, y_pred, class_names, epoch, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))  # 计算混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # 每行归一化为比例

    # 避免除以0导致的 NaN
    cm_normalized = np.nan_to_num(cm_normalized)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2%",  # 显示为百分比
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix (Normalized %) - Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 计算每个类别的评估指标
def calculate_metrics(preds, labels, num_classes=3):
    preds = preds.cpu().numpy().flatten()
    labels = labels.cpu().numpy().flatten()


    iou_list, precision_list, recall_list, dice_list, f1_list = [], [], [], [], []
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        label_cls = (labels == cls)

        TP = np.logical_and(pred_cls, label_cls).sum()
        FP = np.logical_and(pred_cls, ~label_cls).sum()
        FN = np.logical_and(~pred_cls, label_cls).sum()

        union = np.logical_or(pred_cls, label_cls).sum()
        iou = TP / (union + 1e-6)
        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        dice = 2 * TP / (2 * TP + FP + FN + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        iou_list.append(iou)
        precision_list.append(precision)
        recall_list.append(recall)
        dice_list.append(dice)
        f1_list.append(f1)

    return iou_list, precision_list, recall_list, dice_list, f1_list
def calculate_macro_average(iou, precision, recall, dice, f1):
    return (
        np.mean(iou),
        np.mean(precision),
        np.mean(recall),
        np.mean(dice),
        np.mean(f1)
    )

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

for epoch in range(epochs):
    start_time = time.time()
    model.train()
    train_loss = 0
    loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', leave=False)
    for X, Y in loop:
        optimizer.zero_grad()
        X = X.to(device)  # 转移到 GPU
        Y = Y.to(device)  # 转移到 GPU
        X = X.reshape(-1, X.size(2), X.size(3), X.size(4), X.size(5))
        X = X.permute(0, 4, 1, 2, 3)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        outputs = model(X)
        Y = Y.reshape(-1, Y.size(2), Y.size(3), Y.size(4), Y.size(5))
        Y = Y.permute(0, 4, 1, 2, 3)
        Y = Y.squeeze(1)  # 去掉多余的通道维度，变为 [batch_size, depth, height, width]
        Y = Y.long()

        loss = criterion(outputs, Y)
        loss.backward()
        torch.cuda.empty_cache() 
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    # 记录当前epoch的训练时长
    epoch_time = time.time() - start_time  # 计算训练时长
    print(f"Epoch {epoch + 1}/{epochs}, 训练损失: {train_loss:.4f}, 训练时长: {epoch_time:.2f}秒")
    param_count = count_parameters(model) / 1e6
    print(f"模型参数量: {param_count:.2f}M")
    # 记录训练损失和时长到日志文件
    with open("./log/SeismicTUMSAA/train-lossseismicTUMSAA损失优化.txt", mode='a') as file:
        file.write(f'{epoch + 1}\t{train_loss:.4f}\t{epoch_time:.2f}\n')
        file.flush()  # 确保数据写入文件     
    # 在验证集上进行评估
    model.eval()
    val_loss = 0
    val_accuracy = 0
    val_recall = np.zeros(3)
    val_precision = np.zeros(3)
    val_iou = np.zeros(3)
    val_dice = np.zeros(3)
    val_f1 = np.zeros(3)

    val_road_precision = 0  # 河道的宏平均
    val_road_recall = 0
    val_road_f1 = 0
    val_road_iou = 0
    val_road_dice = 0

    val_cave_precision = 0  # 溶洞的宏平均
    val_cave_recall = 0
    val_cave_f1 = 0
    val_cave_iou = 0
    val_cave_dice = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, Y in valid_loader:
            X = X.to(device)
            Y = Y.to(device)
            X = X.reshape(-1, X.size(2), X.size(3), X.size(4), X.size(5))
            X = X.permute(0, 4, 1, 2, 3)
            outputs = model(X)
            Y = Y.reshape(-1, Y.size(2), Y.size(3), Y.size(4), Y.size(5))
            Y = Y.permute(0, 4, 1, 2, 3)
            # Y = Y.long()
            preds = torch.argmax(outputs, dim=1)  # [B, D, H, W]

            # Ground truth 标签
            Y = Y.squeeze(1).long()               # [B, D, H, W]

            # 添加到列表中
            all_preds.append(preds.cpu().numpy().flatten())
            all_labels.append(Y.cpu().numpy().flatten())
            # print("📊 preds unique:", np.unique(all_preds, return_counts=True))
            # print("📊 labels unique:", np.unique(all_labels, return_counts=True))
            loss = criterion(outputs, Y)
            val_loss += loss.item()
            # # preds = torch.argmax(outputs, dim=1)
            # all_preds.append(outputs.cpu().numpy())
            # all_labels.append(Y.cpu().numpy())

    val_loss /= len(valid_loader)

    all_preds_flat = np.concatenate(all_preds)
    all_labels_flat = np.concatenate(all_labels)

    # === 计算指标 ===
    iou_per_class, precision_per_class, recall_per_class, dice_per_class, f1_per_class = calculate_metrics(
        torch.tensor(all_preds_flat), torch.tensor(all_labels_flat), num_classes=3
    )
    macro_iou, macro_precision, macro_recall, macro_dice, macro_f1 = calculate_macro_average(
        iou_per_class, precision_per_class, recall_per_class, dice_per_class, f1_per_class
    )


    # === 输出评估结果 ===
    print(f"Epoch {epoch+1}/{epochs}, 验证损失: {val_loss:.4f}")
    print(f"[背景(类别0)] 精度: {precision_per_class[0]:.4f}, 召回率: {recall_per_class[0]:.4f}, F1: {f1_per_class[0]:.4f}, IoU: {iou_per_class[0]:.4f}")
    print(f"[河道(类别1)] 精度: {precision_per_class[1]:.4f}, 召回率: {recall_per_class[1]:.4f}, F1: {f1_per_class[1]:.4f}, IoU: {iou_per_class[1]:.4f}")
    print(f"[溶洞(类别2)] 精度: {precision_per_class[2]:.4f}, 召回率: {recall_per_class[2]:.4f}, F1: {f1_per_class[2]:.4f}, IoU: {iou_per_class[2]:.4f}")
    print(f"[宏平均] 精度: {macro_precision:.4f}, 召回率: {macro_recall:.4f}, F1: {macro_f1:.4f}, IoU: {macro_iou:.4f}")

    # === 保存到日志文件 ===
    with open("./log/SeismicTUMSAA/val-lossseismicTUMSAA损失优化.txt", mode='a') as file:
        file.write(f"\nEpoch {epoch+1}\n")
        file.write(f"验证损失: {val_loss:.4f}\n")
        file.write(f"[背景(类别0)] 精度: {precision_per_class[0]:.4f}, 召回率: {recall_per_class[0]:.4f}, F1: {f1_per_class[0]:.4f}, IoU: {iou_per_class[0]:.4f}\n")
        file.write(f"[河道(类别1)] 精度: {precision_per_class[1]:.4f}, 召回率: {recall_per_class[1]:.4f}, F1: {f1_per_class[1]:.4f}, IoU: {iou_per_class[1]:.4f}\n")
        file.write(f"[溶洞(类别2)] 精度: {precision_per_class[2]:.4f}, 召回率: {recall_per_class[2]:.4f}, F1: {f1_per_class[2]:.4f}, IoU: {iou_per_class[2]:.4f}\n")
        file.write(f"[宏平均] 精度: {macro_precision:.4f}, 召回率: {macro_recall:.4f}, F1: {macro_f1:.4f}, IoU: {macro_iou:.4f}\n")
        file.flush()

    # 调整学习率
    scheduler.step(val_loss)
    # ✅ 混淆矩阵绘图
    plot_confusion_matrix(
        y_true=all_labels_flat,
        y_pred=all_preds_flat,
        class_names=['背景', '河道', '溶洞'],
        epoch=epoch + 1,
        save_path=f"./log/SeismicTUMSAA/SeismicTUMSAA损失优化_epoch{epoch+1}.png"
    )

    # 保存模型检查点
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, filename=f"/root/autodl-tmp/modelseismicTU-MSAA/SeismicTUMSAA损失优化.{epoch + 1:02d}.pth")

    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

writer.close() 
