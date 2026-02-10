import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset
from data_utilities import load_compact_pkl_dataset
from scipy.signal import stft
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
import math
from SEINet import SEINet_Base_Prototype  # 你的可学习原型模型
from SEILoss import LossPrototypes        # 保留原型分类 loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ntx = 7
nrx = 5

# ===============================
# 数据处理
# ===============================
def data_load(dataset_path="../dataset/", dataset_name="SingleDay"):
    dataset = load_compact_pkl_dataset(dataset_path, dataset_name)
    data = dataset["data"]
    print("DataSet Load Successfully")
    return data

def iq_fft(iq_data: np.ndarray, n_fft=32, window="hann", fs=1.0):
    x = iq_data[:, 0] + 1j * iq_data[:, 1]
    _, _, Zxx = stft(x, fs=fs, window=window, nperseg=n_fft, return_onesided=False, boundary=None, padded=False)        # type: ignore
    return Zxx, np.abs(Zxx)

def pad_time_axis(Zxx, target_time=32):
    _, time = Zxx.shape
    pad_width = target_time - time
    Z_pad = np.pad(Zxx, pad_width=((0,0),(0,pad_width)), mode='constant', constant_values=0)
    return Z_pad, np.abs(Z_pad)

def sftf_to_3_channels(Zxx_32x32):
    x = torch.from_numpy(Zxx_32x32).float().unsqueeze(0)
    x = x.repeat(3,1,1).unsqueeze(0)
    return x

def data_load_prepare(data, ntx=7, nrx=5, batch_size=64):
    data_inputs = []
    data_labels = []
    for txi in range(ntx):
        for rxi in range(nrx):
            for num in range(800):
                iq_data = data[txi][rxi][0][1][num]
                _, abs_zxx = iq_fft(iq_data)
                _, z_abs_32x32 = pad_time_axis(abs_zxx)
                tensor_data = sftf_to_3_channels(z_abs_32x32)
                data_inputs.append(tensor_data)
                data_labels.append([txi, rxi])
    data_inputs = torch.cat(data_inputs, dim=0)
    data_labels = torch.tensor(data_labels)
    X_train, X_temp, y_train, y_temp = train_test_split(data_inputs, data_labels, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)
    return train_loader, val_loader, test_loader

# ===============================
# ProtoNCE 损失
# ===============================
class ProtoNCE(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, prototypes, labels):
        """
        features: (B, D) 样本 embedding
        prototypes: (C, D) 原型权重
        labels: (B,)
        """
        features = F.normalize(features, dim=1)
        prototypes = F.normalize(prototypes, dim=1)
        sim_matrix = torch.matmul(features, prototypes.T) / self.temperature  # (B, C)
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

# ===============================
# 训练函数
# ===============================
def train_model(model, train_loader, val_loader, num_epochs=50, lambda_protoNCE=0.1, lr=1e-3, save_path="./model/base_protoNCE.pth"):
    device = next(model.parameters()).device
    proto_criterion = LossPrototypes()
    proto_nce_criterion = ProtoNCE()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            tx_labels = labels[:,0]

            optimizer.zero_grad()
            features, cosine = model(inputs)

            # 原型分类 loss
            loss_proto = proto_criterion(cosine, tx_labels)
            # ProtoNCE loss
            loss_proto_nce = proto_nce_criterion(features, model.prototype_layer.weight, tx_labels)

            loss = loss_proto + lambda_protoNCE * loss_proto_nce
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)

        train_loss = total_loss / len(train_loader.dataset)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                tx_labels = labels[:,0]

                features, cosine = model(inputs)
                loss_proto = proto_criterion(cosine, tx_labels)
                loss_proto_nce = proto_nce_criterion(features, model.prototype_layer.weight, tx_labels)
                loss = loss_proto + lambda_protoNCE * loss_proto_nce
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✔ Model saved (val loss={best_val_loss:.4f})")

# ===============================
# 测试函数
# ===============================
def test_model(model, test_loader):
    device = next(model.parameters()).device
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            tx_labels = labels[:,0]
            _, cosine = model(inputs)
            preds = torch.argmax(cosine, dim=1)
            correct += (preds == tx_labels).sum().item()
            total += tx_labels.size(0)
    acc = 100.0 * correct / total
    print(f"Test Accuracy (TX): {acc:.2f}%")
    return acc

# ===============================
# 主程序
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lambda_protoNCE", type=float, help="weight of ProtoNCE loss")
    args = parser.parse_args()

    data = data_load()
    train_loader, val_loader, test_loader = data_load_prepare(data, ntx=ntx, nrx=nrx, batch_size=4096)

    model = SEINet_Base_Prototype(num_classes=ntx).to(device)

    train_model(model, train_loader, val_loader, num_epochs=50, lambda_protoNCE=args.lambda_protoNCE, save_path="./model/base_protoNCE.pth")
    test_model(model, test_loader)
