
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
from utils import DataGenerator
import time
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim import Adam

def calculate_metrics(preds, labels, num_classes=3):
    preds = preds.flatten()
    labels = labels.flatten()
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
    
# 三线性插值平滑权重函数保持不变
def create_trilinear_weights(size, overlap):
    weights = np.ones((size, size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            for k in range(size):
                wi = 1.0
                wj = 1.0
                wk = 1.0
                if i < overlap:
                    wi = (i + 1) / (overlap + 1)
                elif i >= size - overlap:
                    wi = (size - i) / (overlap + 1)
                if j < overlap:
                    wj = (j + 1) / (overlap + 1)
                elif j >= size - overlap:
                    wj = (size - j) / (overlap + 1)
                if k < overlap:
                    wk = (k + 1) / (overlap + 1)
                elif k >= size - overlap:
                    wk = (size - k) / (overlap + 1)
                weights[i, j, k] = wi * wj * wk
    return torch.from_numpy(weights)

def get_start_indices(length, window, overlap):
    """返回起点列表，步长 = window - overlap；并保证最后一个块覆盖到末尾。"""
    stride = max(1, window - overlap)
    idxs = list(range(0, max(0, length - window + 1), stride))
    if len(idxs) == 0 or idxs[-1] != length - window:
        idxs.append(length - window)
    return idxs

def goFakeValidation(model, fname, output_dir):
    # 定义路径
    seisPath = "/root/autodl-fs/ChannelKarst/seismic/"
    lxpath = "/root/autodl-fs/ChannelKarst/label/"
    predPath = "/root/autodl-fs/ChannelKarst/px/"
    
    # 原始数据体素维度 (D, H, W)
    n1, n2, n3 = 256, 256, 256
    input_shape = (n1, n2, n3)  # 修改为256x256x256
    
    # 3D patch 配置
    patch_d = patch_h = patch_w = 128
    overlap = 2  # 每个维度的重叠体素数
    
    try:
        gx = loadData(n1, n2, n3, seisPath, fname + '.dat')
        gx = np.reshape(gx, (1, 1, n1, n2, n3))  # 修改为PyTorch格式

        print("✅ Data loaded. Shape:", gx.shape)  # [1,1,D,H,W]
        
        # # 加载标签数据
        # label
        lx = loadData1(n1, n2, n3, lxpath, fname + '.dat')
        ls = np.reshape(lx, (1, 1, n1, n2, n3))
        print("📊 ls unique:", np.unique(ls, return_counts=True))
        
        # 映射标签值到0,1,2
        # 根据标签值的分布，将最小值映射为0，中间值映射为1，最大值映射为2
        unique_vals = np.unique(lx)
        sorted_vals = np.sort(unique_vals)
        
        label_mapping = {
            sorted_vals[0]: 0,  # 最小值映射为0（背景）
            sorted_vals[1]: 1,  # 中间值映射为1（河道）
            sorted_vals[2]: 2   # 最大值映射为2（溶洞）
        }
        
        labels_mapped = np.copy(lx)
        for src_val, dst_val in label_mapping.items():
            labels_mapped[lx == src_val] = dst_val
        print("📊 Mapped labels unique:", np.unique(labels_mapped, return_counts=True))
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    gx = torch.from_numpy(gx).to(device)  # [1,1,D,H,W]
    
    # 输出张量与权重图（按类别数 3）
    B, C, D, H, W = gx.shape
    num_classes = 3
    output = torch.zeros((B, num_classes, D, H, W), device=device)
    weight_map = torch.zeros((B, 1, D, H, W), device=device)
    
    # 预计算平滑权重 (三线性核)，广播到 [B,1,d,h,w]
    weight = create_trilinear_weights(patch_d, overlap).to(device)  # [128,128,128]
    weight = weight.unsqueeze(0).unsqueeze(0)  # [1,1,128,128,128]
    
    # 计算三个维度的起点索引
    d_starts = get_start_indices(D, patch_d, overlap)
    h_starts = get_start_indices(H, patch_h, overlap)
    w_starts = get_start_indices(W, patch_w, overlap)
    
    start = time.time()
    print("🚀 Running 3D sliding-window prediction with trilinear smoothing...")
    print(f" D starts: {d_starts}")
    print(f" H starts: {h_starts}")
    print(f" W starts: {w_starts}")
    
    try:
        with torch.no_grad():
            for ds in d_starts:
                de = ds + patch_d
                for hs in h_starts:
                    he = hs + patch_h
                    for ws in w_starts:
                        we = ws + patch_w
                        # 取 3D patch: [B, C, 128,128,128]
                        patch = gx[:, :, ds:de, hs:he, ws:we]
                        
                        # 模型前向: 期望输出 [B, num_classes, 128,128,128]
                        pred_patch = model(patch)
                        
                        # 三线性权重平滑
                        pred_patch = pred_patch * weight
                        
                        # 累加预测与权重
                        output[:, :, ds:de, hs:he, ws:we] += pred_patch
                        weight_map[:, :, ds:de, hs:he, ws:we] += weight
            
            # 避免除零（通常不会为零）
            weight_map = torch.clamp(weight_map, min=1e-6)
            output /= weight_map  # 广播到类别维
            
            # 取类别标签
            preds = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            
            print("✅ Prediction completed in {:.2f}s.".format(time.time() - start))
            print("Prediction shape:", preds.shape)  # (D,H,W)
            
            # 确保输出目录存在
            os.makedirs(predPath, exist_ok=True)
            
            # 保存预测结果
            preds.astype(np.float32).tofile(os.path.join(predPath, f"91{fname}_pred.dat"))
            (preds == 1).astype(np.float32).tofile(os.path.join(predPath, f"91{fname}_river.dat"))
            (preds == 2).astype(np.float32).tofile(os.path.join(predPath, f"91{fname}_cave.dat"))
            
            print("✅ Saved river and cave masks.")

            # ===== 计算宏平均指标 =====
            # 使用映射后的标签数据
            labels = labels_mapped
            
            # 展平预测和标签
            preds_flat = preds.flatten()
            labels_flat = labels.flatten()
            
            # 3. 计算指标
            iou_per_class, precision_per_class, recall_per_class, dice_per_class, f1_per_class = calculate_metrics(
                preds_flat, labels_flat, num_classes=3
            )
            
            # 4. 计算宏平均
            macro_iou = np.mean(iou_per_class)
            macro_precision = np.mean(precision_per_class)
            macro_recall = np.mean(recall_per_class)
            macro_dice = np.mean(dice_per_class)
            macro_f1 = np.mean(f1_per_class)
            
            # 5. 打印结果
            print(f"🌐 宏平均: IoU={macro_iou:.4f}, Precision={macro_precision:.4f}, Recall={macro_recall:.4f}, F1={macro_f1:.4f}, Dice={macro_dice:.4f}")
            print("📊 preds unique:", np.unique(preds, return_counts=True))
            print("📊 labels unique:", np.unique(labels, return_counts=True))
            
            # 6. 按类别输出结果
            class_names = ['背景', '河道', '溶洞']
            print(f"{fname}_175")
            for i, cls_name in enumerate(class_names):
                print(
                    f"[{cls_name}(类别{i})] 精度: {precision_per_class[i]:.4f}, 召回率: {recall_per_class[i]:.4f}, "
                    f"F1: {f1_per_class[i]:.4f}, IoU: {iou_per_class[i]:.4f}, Dice: {dice_per_class[i]:.4f}"
                )
            
            # 7. 保存到日志文件
            metrics_log_path = os.path.join(predPath, "91metrics_log.txt")
            with open(metrics_log_path, "a") as file:
                file.write(f"===== 样本 {fname} 评估结果 =====\n")
                file.write(f"宏平均: IoU={macro_iou:.4f}, Precision={macro_precision:.4f}, Recall={macro_recall:.4f}, F1={macro_f1:.4f}, Dice={macro_dice:.4f}\n")
                for i, cls_name in enumerate(class_names):
                    file.write(
                        f"[{cls_name}(类别{i})] 精度: {precision_per_class[i]:.4f}, 召回率: {recall_per_class[i]:.4f}, "
                        f"F1: {f1_per_class[i]:.4f}, IoU: {iou_per_class[i]:.4f}, Dice: {dice_per_class[i]:.4f}\n"
                    )
                file.write("\n")
            
            # 8. 绘制混淆矩阵
            plot_confusion_matrix(
                y_true=labels_flat,
                y_pred=preds_flat,
                class_names=class_names,
                epoch=fname,  # 使用文件名作为epoch标识
                save_path=os.path.join(predPath, f"{fname}_TUMSAAconfusion.png")
            )
            print(f"✅ 混淆矩阵已保存: {fname}_TUMSAAconfusion.png")
            
    except Exception as e:
        print(f"❌ Error during prediction: {e}")


        
def loadData(n1, n2, n3, path, fname):
    gx = np.fromfile(path + fname, dtype=np.single)
    gm, gs = np.mean(gx), np.std(gx)
    gx = (gx - gm) / gs
    gx = np.reshape(gx, (n3, n2, n1))
    gx = np.transpose(gx)
    return gx

def loadData1(n1, n2, n3, path, fname):
    lx = np.fromfile(path + fname, dtype=np.int8)
    lm, ls = np.mean(lx), np.std(lx)
    lx = lx - lm
    lx = lx / ls
    lx = np.reshape(lx, (n3, n2, n1))
    lx = np.transpose(lx)
    return lx

import os
import time
import numpy as np
import torch
import torch.nn.functional as F

# 三线性插值平滑权重
def create_trilinear_weights(size, overlap):
    weights = np.ones((size, size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            for k in range(size):
                wi = 1.0
                wj = 1.0
                wk = 1.0
                if i < overlap:
                    wi = (i + 1) / (overlap + 1)
                elif i >= size - overlap:
                    wi = (size - i) / (overlap + 1)
                if j < overlap:
                    wj = (j + 1) / (overlap + 1)
                elif j >= size - overlap:
                    wj = (size - j) / (overlap + 1)
                if k < overlap:
                    wk = (k + 1) / (overlap + 1)
                elif k >= size - overlap:
                    wk = (size - k) / (overlap + 1)
                weights[i, j, k] = wi * wj * wk
    return torch.from_numpy(weights)

import os, time
import numpy as np
import torch

def get_start_indices(length, window, overlap):
    """返回起点列表，步长 = window - overlap；并保证最后一个块覆盖到末尾。"""
    stride = max(1, window - overlap)
    idxs = list(range(0, max(0, length - window + 1), stride))
    if len(idxs) == 0 or idxs[-1] != length - window:
        idxs.append(length - window)
    return idxs

def goJie(model, fname, output_dir):
    fpath = "/root/autodl-fs/"
    # 原始数据体素维度 (D, H, W)
    input_shape = (128, 896, 384)
    # input_shape = (128, 512, 640)
    # input_shape = (128, 256, 256)
    # input_shape = (128, 512, 512)

    # 3D patch 配置
    patch_d = patch_h = patch_w = 128
    overlap = 2  # 每个维度的重叠体素数

    try:
        gx = np.load(os.path.join(fpath, fname)).astype(np.float32).reshape(1, 1, *input_shape)
        print("✅ Data loaded. Shape:", gx.shape)  # [1,1,D,H,W]
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    gx = torch.from_numpy(gx).to(device)  # [1,1,D,H,W]

    # 输出张量与权重图（按类别数 3）
    B, C, D, H, W = gx.shape
    num_classes = 3
    output = torch.zeros((B, num_classes, D, H, W), device=device)
    weight_map = torch.zeros((B, 1, D, H, W), device=device)

    # 预计算平滑权重 (三线性核)，广播到 [B,1,d,h,w]
    weight = create_trilinear_weights(patch_d, overlap).to(device)  # [128,128,128]
    weight = weight.unsqueeze(0).unsqueeze(0)  # [1,1,128,128,128]

    # 计算三个维度的起点索引
    d_starts = get_start_indices(D, patch_d, overlap)
    h_starts = get_start_indices(H, patch_h, overlap)
    w_starts = get_start_indices(W, patch_w, overlap)

    start = time.time()
    print("🚀 Running 3D sliding-window prediction with trilinear smoothing...")
    print(f"   D starts: {d_starts}")
    print(f"   H starts: {h_starts}")
    print(f"   W starts: {w_starts}")

    try:
        with torch.no_grad():
            for ds in d_starts:
                de = ds + patch_d
                for hs in h_starts:
                    he = hs + patch_h
                    for ws in w_starts:
                        we = ws + patch_w

                        # 取 3D patch: [B, C, 128,128,128]
                        patch = gx[:, :, ds:de, hs:he, ws:we]

                        # 模型前向: 期望输出 [B, num_classes, 128,128,128]
                        pred_patch = model(patch)

                        # 若模型返回 logits（一般如此），直接加权融合；需要 softmax 请在这里加
                        # pred_patch = torch.softmax(pred_patch, dim=1)

                        # 三线性权重平滑
                        pred_patch = pred_patch * weight

                        # 累加预测与权重
                        output[:, :, ds:de, hs:he, ws:we] += pred_patch
                        weight_map[:, :, ds:de, hs:he, ws:we] += weight

        # 避免除零（通常不会为零）
        weight_map = torch.clamp(weight_map, min=1e-6)
        output /= weight_map  # 广播到类别维

        # 取类别标签
        preds = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        print("✅ Prediction completed in {:.2f}s.".format(time.time() - start))
        print("Prediction shape:", preds.shape)  # (D,H,W)

        # 保存
        os.makedirs(output_dir, exist_ok=True)
        # np.save(os.path.join(output_dir, "94ChannelTUMSAApredicted_mask.npy"), preds)
        # np.save(os.path.join(output_dir, "94SHUNBEISeismicTUMSAApredicted_mask.npy"), preds)
        np.save(os.path.join(output_dir, "82SeismicTUMSAApredicted_mask82.npy"), preds)

        # 单独保存 river / cave
        river_mask = (preds == 1).astype(np.uint8)
        cave_mask = (preds == 2).astype(np.uint8)
        # np.save(os.path.join(output_dir, "94ChannelTUMSAAriver_mask.npy"), river_mask)
        # np.save(os.path.join(output_dir, "94ChannelTUMSAAcave_mask.npy"), cave_mask)
        # np.save(os.path.join(output_dir, "94SHUNBEISeismicTUMSAAriver_mask.npy"), river_mask)
        # np.save(os.path.join(output_dir, "94SHUNBEISeismicTUMSAAcave_mask.npy"), cave_mask)
        np.save(os.path.join(output_dir, "82SeismicTUMSAAriver_mask82.npy"), river_mask)
        np.save(os.path.join(output_dir, "82SeismicTUMSAAcave_mask82.npy"), cave_mask)
        print("✅ Saved river and cave masks.")

    except Exception as e:
        print(f"❌ Error during prediction: {e}")


if __name__ == '__main__':
    # 初始化模型、损失函数和优化器
    config = configs.get_r50_b16_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionTransformer(config, img_size=128, num_classes=21843, zero_head=False, vis=False).to(device)
    # 1. 加载模型
    checkpoint = torch.load('/root/autodl-tmp/modelseismicTU-MSAA/SeismiccheckpointTUMSAA.91.pth')
    model.load_state_dict(checkpoint['state_dict'])  # 使用 'state_dict' 键
    model.eval()
        
    seisPath = "/root/autodl-fs/ChannelKarst/seismic/"
    lxpath = "/root/autodl-fs/ChannelKarst/label/"
    predPath = "/root/autodl-fs/ChannelKarst/px/"
    # 定义要处理的文件列表
    ks = [140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 
          150, 151, 152, 153, 154, 155, 156, 157, 158, 159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190]
    
    for k in ks:
        fname = str(k)
        print(f"🚀 Processing file: {fname}")
        goFakeValidation(model, fname, predPath)
        print(f"✅ Finished processing {fname}")
        
#     # 2. 运行预测
#     result_files = goJie(
#         model=model,
#         fname="rwp384-896-128.npy",  # 替换为您的数据文件名
#         # fname="SHB4_NORTH_slice640-512-128.npy",  # 替换为您的数据文件名
#         # fname="Parihaka_sliceNew-ZHAO.npy",  # 替换为您的数据文件名
#         # fname="seismic_192-slice4.npy",  # 替换为您的数据文件名
#         # fname="Channel_128_512_512.npy",  # 替换为您的数据文件名

#         output_dir="/root/autodl-fs/results/"  # 替换为您希望保存输出的目录
#     )


# ######输出结果为logit
# import torch
# import numpy as np
# import os
# import time

# # 三线性插值平滑权重
# def create_trilinear_weights(size, overlap):
#     weights = np.ones((size, size, size), dtype=np.float32)
#     for i in range(size):
#         for j in range(size):
#             for k in range(size):
#                 wi = 1.0
#                 wj = 1.0
#                 wk = 1.0
#                 if i < overlap:
#                     wi = (i + 1) / (overlap + 1)
#                 elif i >= size - overlap:
#                     wi = (size - i) / (overlap + 1)
#                 if j < overlap:
#                     wj = (j + 1) / (overlap + 1)
#                 elif j >= size - overlap:
#                     wj = (size - j) / (overlap + 1)
#                 if k < overlap:
#                     wk = (k + 1) / (overlap + 1)
#                 elif k >= size - overlap:
#                     wk = (size - k) / (overlap + 1)
#                 weights[i, j, k] = wi * wj * wk
#     return torch.from_numpy(weights)

# def get_start_indices(length, window, overlap):
#     """返回起点列表，步长 = window - overlap；并保证最后一个块覆盖到末尾。"""
#     stride = max(1, window - overlap)
#     idxs = list(range(0, max(0, length - window + 1), stride))
#     if len(idxs) == 0 or idxs[-1] != length - window:
#         idxs.append(length - window)
#     return idxs

# def goJie(model, fname, output_dir):
#     # 文件路径设置
#     fpath = r"/root/autodl-fs/"
#     n1, n2, n3 = 128, 896, 384  # 输入数据的尺寸
#     input_size = 128
#     overlap = 4
    
#     # 开始计时
#     start_time = time.time()

#     # 加载地震数据
#     try:
#         gx = np.load(fpath + fname)
#         gx = np.reshape(gx, (1, 1, n1, n2, n3))  # 转换为PyTorch格式
#         gx_torch = torch.from_numpy(gx).float()  # 转换为PyTorch张量
#         print("Data loaded successfully. Shape:", gx_torch.shape)
#     except Exception as e:
#         print(f"Error loading data: {str(e)}")
#         return

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device).eval()

#     gx_torch = gx_torch.to(device)  # [1, 1, 384, 896, 128]

#     # 获取各个维度的起始位置
#     d_starts = get_start_indices(gx_torch.shape[2], input_size, overlap)
#     h_starts = get_start_indices(gx_torch.shape[3], input_size, overlap)
#     w_starts = get_start_indices(gx_torch.shape[4], input_size, overlap)

#     # 输出张量与权重图（按类别数 3）
#     B, C, D, H, W = gx_torch.shape
#     num_classes = 3
#     output = torch.zeros((B, num_classes, D, H, W), device=device)
#     weight_map = torch.zeros((B, 1, D, H, W), device=device)

#     # 创建平滑权重 (三线性核)，广播到 [B,1,d,h,w]
#     weight = create_trilinear_weights(input_size, overlap).to(device)  # [128,128,128]
#     weight = weight.unsqueeze(0).unsqueeze(0)  # [1,1,128,128,128]
#     # 初始化 logits_river 和 logits_cave
#     logits_river = torch.zeros((B, 1, D, H, W), device=device)
#     logits_cave = torch.zeros((B, 1, D, H, W), device=device)
#     # 开始分块预测
#     try:

#         with torch.no_grad():
#             for ds in d_starts:
#                 de = ds + input_size
#                 for hs in h_starts:
#                     he = hs + input_size
#                     for ws in w_starts:
#                         we = ws + input_size

#                         # 取 3D patch: [B, C, 128,128,128]
#                         patch = gx_torch[:, :, ds:de, hs:he, ws:we]

#                         # 模型前向: 期望输出 [B, num_classes, 128,128,128]
#                         pred_patch = model(patch)

#                         # 三线性权重平滑
#                         pred_patch = pred_patch * weight

#                         # 累加预测与权重
#                         output[:, :, ds:de, hs:he, ws:we] += pred_patch
#                         weight_map[:, :, ds:de, hs:he, ws:we] += weight

#                         # 只保存河道与溶洞的 logits
#                         logits_river[:, :, ds:de, hs:he, ws:we] += pred_patch[0, 1, :, :, :]  # 取河道类的 logits
#                         logits_cave[:, :, ds:de, hs:he, ws:we] += pred_patch[0, 2, :, :, :]  # 取溶洞类的 logits

#         # 避免除零（通常不会为零）
#         weight_map = torch.clamp(weight_map, min=1e-6)
#         output /= weight_map  # 广播到类别维

#         # 获取每个体素的类别，输出形状：[1, 384, 896, 128]
#         preds = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

#         print("✅ Prediction completed in {:.2f}s.".format(time.time() - start_time))
#         print("Prediction shape:", preds.shape)  # (D,H,W)
#         print(f"Channel: {logits_river.shape}")
#         print(f"Cave: {logits_cave.shape}")
        
#         # 保存
#         os.makedirs(output_dir, exist_ok=True)

#         # 去掉所有多余的维度，将 shape 转换为 (384, 896, 128)
#         logits_river = logits_river.squeeze().cpu().numpy()  # 这会去掉所有维度为 1 的轴
#         logits_cave = logits_cave.squeeze().cpu().numpy()  # 同样去掉所有维度为 1 的轴

#         print(f"Channel111: {logits_river.shape}")  # 期望输出 (384, 896, 128)
#         print(f"Cave111: {logits_cave.shape}")  # 期望输出 (384, 896, 128)


#         # 保存河道概率
#         channel_file = os.path.join(output_dir, "Channel_prob.npy")
#         np.save(channel_file, logits_river)
#         print(f"Channel probability saved to {channel_file}")

#         # 保存溶洞概率
#         cave_file = os.path.join(output_dir, "Cave_prob.npy")
#         np.save(cave_file, logits_cave)
#         print(f"Cave probability saved to {cave_file}")
        
#         # 将输出形状转为 (D, H, W) (每个体素的最大logit类别)
#         preds_final = np.moveaxis(preds, 0, -1)  # 转换为 (D, H, W) 的形状

#         # 保存最终分类结果（优先溶洞 -> 河道 -> 背景）
#         combined = np.zeros_like(preds_final, dtype=np.uint8)

#         # 先标记溶洞区域为 2
#         combined[preds_final == 2] = 2    # 溶洞 (概率>0.5)

#         # 然后标记河道区域为 1，只有当溶洞区域为 0 时才标记
#         combined[(preds_final == 1) & (combined == 0)] = 1 # 河道 (概率>0.5)

#         # 保存最终的分类结果
#         combined_file = os.path.join(output_dir, "Combined_classification.npy")
#         np.save(combined_file, combined)
#         print(f"Combined classification saved to {combined_file}")

#     except Exception as e:
#         print(f"Error during prediction: {str(e)}")
#         return

# if __name__ == '__main__':
#     # 初始化模型、损失函数和优化器
#     config = configs.get_r50_b16_config()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = VisionTransformer(config, img_size=128, num_classes=21843, zero_head=False, vis=False).to(device)
    
#     # 1. 加载模型
#     checkpoint = torch.load('/root/autodl-tmp/modelseismicTU-MSAA/SeismiccheckpointTUMSAA.94.pth')
#     model.load_state_dict(checkpoint['state_dict'])  # 使用 'state_dict' 键
#     model.eval()

#     # 2. 运行预测
#     result_files = goJie(
#         model=model,
#         fname="rwp384-896-128.npy",  # 替换为您的数据文件名
#         output_dir="/root/autodl-fs/results/"  # 替换为您希望保存输出的目录
#     )
