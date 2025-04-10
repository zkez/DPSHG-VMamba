import os
import torch
import time
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score


import sys
sys.path.append("./")
from data.dataloader import LungNoduleDataset
from models.mamba import SpatiotemporalMamba
from models.utils import Runtime_Observer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

csv_path = '/home/zk/MICCAI/newmainroi.csv'  
data_dir = '/home/zk/MICCAI/roiresize'
text_csv_path = '/home/zk/MICCAI/scale_information.csv'
csv_data = pd.read_csv(csv_path)
text_data = pd.read_csv(text_csv_path)
    
# 创建 KFold 对象
#kf = KFold(n_splits=10, shuffle=True, random_state=42)
# 将 subject_ids 转换为列表
subject_ids = csv_data['Subject ID'].unique()
model = SpatiotemporalMamba(
    in_channels=1,
    spatial_size=(64, 64),
    temporal_size=16,
    num_classes=2,
    stage_depths=[2, 4, 8, 16],
    stage_dims=[24, 48, 96, 192]
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5,0.999))
num_epochs=100

# 训练和验证集划分
train_ids, val_ids = train_test_split(subject_ids, test_size=0.15, random_state=8)  # 80% 训练集，20% 验证集

# 划分训练集和验证集
train_data = csv_data[csv_data['Subject ID'].isin(train_ids)]
val_data = csv_data[csv_data['Subject ID'].isin(val_ids)]

# 创建数据集
train_dataset = LungNoduleDataset(train_data, data_dir, text_data, normalize=True)
val_dataset = LungNoduleDataset(val_data, data_dir, text_data, normalize=True)

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# 初始化模型、损失函数和优化器
best_val_loss = float('inf')  # 最优验证集损失
train_losses, val_losses = [], []  # 记录损失
val_accuracies, val_aucs, val_f1_scores = [], [], []  # 记录验证指标

# 损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
if not os.path.exists(f"debug"):
    os.makedirs(f"debug")
observer = Runtime_Observer(log_dir=f"debug", device=device, name="debug", seed=42)
num_params = 0
for p in model.parameters():
    if p.requires_grad:
        num_params += p.numel()
print("\n===============================================\n")
print("model parameters: " + str(num_params))
print("\n===============================================\n")
# 训练过程
def train_model(model, train_loader, val_loader, device, optimizer, criterion, num_epochs=10, scheduler=None):
    train_losses = []
    train_accuracies = []
    val_losses = []
    auc_scores = []
    accuracies = []
    epoch_steps = len(train_loader)
    best_val_loss = float('inf')
    start_time = time.time()
    observer.log("start training\n")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # 训练阶段
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for batch in train_loader:
                # 数据加载
                t0, t1, t2, labels = batch['T0_image'].to(device), batch['T1_image'].to(device), batch['T2_image'].to(device), batch['label'].to(device)
                t0 = t0.unsqueeze(1)
                t1 = t1.unsqueeze(1)
                t2 = t2.unsqueeze(1)
                
                labels = batch['label'].to(device)  # 保持原始数据类型
                table_info = batch['table_info'].float().to(device)

                # 前向传播
                optimizer.zero_grad()
                outputs = model(t0, t1, t2, table_info)

                # 损失计算
                labels_one_hot = torch.zeros_like(outputs)
                labels_one_hot.scatter_(1, labels.unsqueeze(1).long(), 1)
                loss = criterion(outputs, labels_one_hot)

                # 反向传播
                loss.backward()
                optimizer.step()

                # 统计信息
                running_loss += loss.item() * t0.size(0)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                correct_predictions += (preds.argmax(dim=1) == labels).sum().item()
                total_samples += labels.size(0)

                pbar.set_postfix({
                    'Loss': running_loss / total_samples,
                    'Acc': correct_predictions / total_samples
                })
                pbar.update()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # 数据加载
                t0, t1, t2, labels = batch['T0_image'].to(device), batch['T1_image'].to(device), batch['T2_image'].to(device), batch['label'].to(device)
                t0 = t0.unsqueeze(1)
                t1 = t1.unsqueeze(1)
                t2 = t2.unsqueeze(1)
                
                labels = batch['label'].to(device)  # 保持原始数据类型
                table_info = batch['table_info'].float().to(device)

                # 前向传播
                outputs = model(t0, t1, t2, table_info)

                # 损失计算
                labels_one_hot = torch.zeros_like(outputs)
                labels_one_hot.scatter_(1, labels.unsqueeze(1).long(), 1)
                loss = criterion(outputs, labels_one_hot)
                val_loss += loss.item() * t0.size(0)

                # 统计信息
                probs = torch.sigmoid(outputs)
                preds = probs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                # 收集AUC数据
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算指标
        train_loss = running_loss / len(train_loader.dataset)
        # train_acc = correct_predictions / total_samples
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # 计算AUC
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5  # 处理只有单一类别的情况
        
        # 记录结果
        train_losses.append(train_loss)
        # train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        accuracies.append(val_acc)
        auc_scores.append(auc)

        # 早停和模型保存逻辑
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        # 学习率调度
        if scheduler:
            scheduler.step(val_loss)

        # 日志输出
        observer.log(f"Epoch {epoch+1}/{num_epochs}")
        observer.log(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        observer.log(f"Val Acc: {val_acc:.4f} | AUC: {auc:.4f}\n")

    # 最终保存
    torch.save(model.state_dict(), 'final_model.pth')
    observer.log(f"Training complete. Time: {time.time()-start_time:.2f}s")
    
    return train_losses, val_losses, auc_scores, accuracies

# 训练模型
train_losses, val_losses, auc_scores, _= train_model(model, train_loader, val_loader, device, optimizer, criterion, num_epochs=100)

