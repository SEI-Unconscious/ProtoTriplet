import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.signal import stft
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from datetime import datetime
import os

# 导入你的自定义模块
from data_utilities import load_compact_pkl_dataset
from SEINet import SEINet_DANN_FewShot # 确保你已将此模型放入 SEINet.py

# =================配置区域=================
ntx = 7
nrx = 5 # 注意：根据你的逻辑，data_load_prepare 会读取索引为 nrx (即第6个) 的接收机作为目标域
few_shot_nums = [0, 1, 2, 3, 5, 10, 20, 50]
current_date = datetime.now().strftime("%Y_%m_%d")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_save_path = "./model/"
result_save_path = "../results/"

if not os.path.exists(model_save_path): os.makedirs(model_save_path)
if not os.path.exists(result_save_path): os.makedirs(result_save_path)

# =================数据处理函数=================
def iq_fft(iq_data: np.ndarray, n_fft=32):
    x = iq_data[:, 0] + 1j * iq_data[:, 1]
    _, _, Zxx = stft(x, fs=1.0, window="hann", nperseg=n_fft, return_onesided=False, boundary=None, padded=False)       # type: ignore
    return np.abs(Zxx)

def pad_time_axis(abs_zxx, target_time=32):
    freq, time = abs_zxx.shape
    pad_width = target_time - time
    z_pad = np.pad(abs_zxx, pad_width=((0, 0), (0, pad_width)), mode='constant', constant_values=0)
    return z_pad

def sftf_to_3_channels(z_abs_32x32):
    x = torch.from_numpy(z_abs_32x32).float().unsqueeze(0)
    x = x.repeat(3, 1, 1).unsqueeze(0) # (1, 3, 32, 32)
    return x

def data_load_prepare(data, ntx=7, target_rxi=5):
    """
    专门提取目标接收机的数据进行测试/微调
    """
    data_cross_inputs = []
    data_cross_labels = []
    print(f"Preparing data for Target Receiver index: {target_rxi}")
    for txi in range(ntx):
        # 假设 data[txi][target_rxi] 存在
        for num in range(800):
            iq_data = data[txi][target_rxi][0][1][num]
            abs_zxx = iq_fft(iq_data)
            z_abs_32x32 = pad_time_axis(abs_zxx, target_time=32)
            tensor_data = sftf_to_3_channels(z_abs_32x32)
            data_cross_inputs.append(tensor_data)
            data_cross_labels.append([txi, 0])

    data_cross_inputs = torch.cat(data_cross_inputs, dim=0)
    data_cross_labels = torch.tensor(data_cross_labels)
    return data_cross_inputs, data_cross_labels

def get_few_shot_loaders(data_inputs, data_labels, shot, batch_size=128):
    """
    根据 shot 数量划分训练、验证和测试集
    """
    if shot > 0:
        # 每个类别采样 shot 个样本作为训练集
        # 计算总体训练集比例 (shot * ntx) / total_samples
        train_ratio = (shot * ntx) / len(data_inputs)
        X_train, X_remain, y_train, y_remain = train_test_split(
            data_inputs, data_labels, train_size=train_ratio, stratify=data_labels[:, 0], random_state=42
        )
        # 剩下的平分给验证和测试
        X_val, X_test, y_val, y_test = train_test_split(
            X_remain, y_remain, test_size=0.5, random_state=42
        )
        
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    else:
        # 0-shot 模式：直接全部作为测试集
        train_loader, val_loader = None, None
        X_test, y_test = data_inputs, data_labels

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)
    return train_loader, val_loader, test_loader

# =================训练与测试引擎=================
def cross_model_train(model, train_loader, val_loader, epochs=50):
    if train_loader is None: return model
    
    criterion = nn.CrossEntropyLoss()
    # 注意：只给优化器传递 classifier 的参数，节省计算
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels[:, 0].to(device)
            optimizer.zero_grad()
            _, outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels[:, 0].to(device)
                _, outputs = model(inputs)
                val_loss += criterion(outputs, labels).item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 临时保存该 shot 下的最佳模型
            torch.save(model.state_dict(), f"{model_save_path}temp_best_fewshot.pth")
            
    model.load_state_dict(torch.load(f"{model_save_path}temp_best_fewshot.pth"))
    return model

def cross_model_test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels[:, 0].to(device)
            _, outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# =================主函数=================
def main():
    print(f"Starting DANN Few-Shot Experiment - {current_date}")
    
    # 1. 加载数据
    dataset = load_compact_pkl_dataset("../dataset/", "SingleDay")
    data = dataset["data"]
    # 提取目标接收机数据 (例如第6个接收机 rxi=5)
    data_inputs, data_labels = data_load_prepare(data, ntx=ntx, target_rxi=nrx)
    
    results = []

    # 2. 遍历小样本数量
    for shot in few_shot_nums:
        print(f"\n>>> Evaluating {shot}-shot...")
        
        train_loader, val_loader, test_loader = get_few_shot_loaders(data_inputs, data_labels, shot)
        
        # 3. 初始化模型 (内部自动加载 DANN backbone 并冻结)
        model = SEINet_DANN_FewShot(
            num_classes=ntx, 
            dann_model_path=os.path.join(model_save_path, "dann_model_best.pth")
        ).to(device)
        
        # 4. 训练 (仅微调分类器)
        if shot > 0:
            model = cross_model_train(model, train_loader, val_loader, epochs=60)
        
        # 5. 测试
        acc = cross_model_test(model, test_loader)
        print(f"Result for {shot}-shot: Accuracy = {acc:.2f}%")
        results.append(acc)

    # 6. 保存结果
    df = pd.DataFrame({
        'few_shot_nums': few_shot_nums,
        'dann_few_shot_accuracys': results
    })
    csv_name = f'dann_few_shot_results_{current_date}.csv'
    df.to_csv(os.path.join(result_save_path, csv_name), index=False)
    print(f"\nAll experiments finished. Results saved to {csv_name}")

if __name__ == '__main__':
    main()