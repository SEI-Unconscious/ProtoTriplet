import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# 导入你提供的工具函数和模型
from data_utilities import load_compact_pkl_dataset
from SEINet import DANN_SEINet  
from scipy.signal import stft

# 配置参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NTX = 7
NRX = 5

# --- 沿用你的预处理函数 ---
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
    x = x.repeat(3, 1, 1) # (3, 32, 32)
    return x

# --- 直接 Copy 你的 data_load_prepare 并微调 ---
def data_load_prepare(data, ntx=7, nrx=5, batch_size=128):
    data_inputs = []
    data_labels = []
    print("Preprocessing data...")
    for txi in range(ntx):
        for rxi in range(nrx):
            for num in range(800):
                iq_data = data[txi][rxi][0][1][num]
                abs_zxx = iq_fft(iq_data)
                z_abs_32x32 = pad_time_axis(abs_zxx, target_time=32)
                tensor_data = sftf_to_3_channels(z_abs_32x32)
                data_inputs.append(tensor_data.unsqueeze(0)) # 保持维度 (1, 3, 32, 32)
                data_labels.append([txi, rxi])

    data_inputs = torch.cat(data_inputs, dim=0)
    data_labels = torch.tensor(data_labels)
    
    X_train, X_temp, y_train, y_temp = train_test_split(data_inputs, data_labels, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)
    return train_loader, val_loader, test_loader

# --- 补全的测试函数 ---
def test_dann(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, tx_labels = inputs.to(DEVICE), labels[:, 0].to(DEVICE)
            # 测试时 alpha 不重要，因为我们只看 tx_logits
            tx_logits, _ = model(inputs, alpha=0.0)
            _, predicted = torch.max(tx_logits.data, 1)
            total += tx_labels.size(0)
            correct += (predicted == tx_labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# --- 训练函数 ---
def train_dann(model, train_loader, val_loader, epochs=50):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        # 动态 alpha 调度
        p = epoch / epochs
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        running_tx_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            tx_labels = labels[:, 0].to(DEVICE)
            rx_labels = labels[:, 1].to(DEVICE)

            optimizer.zero_grad()
            tx_logits, rx_logits = model(inputs, alpha=alpha)
            
            loss_tx = criterion(tx_logits, tx_labels)
            loss_rx = criterion(rx_logits, rx_labels)
            
            loss = loss_tx + loss_rx
            loss.backward()
            optimizer.step()
            running_tx_loss += loss_tx.item()

        val_acc = test_dann(model, val_loader)
        print(f"Epoch {epoch+1}/{epochs} | Alpha: {alpha:.3f} | TX Loss: {running_tx_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "./model/dann_model_best.pth")
    
    return model

if __name__ == "__main__":
    from data_utilities import load_compact_pkl_dataset
    
    # 1. 加载数据
    raw_data = load_compact_pkl_dataset("../dataset/", "SingleDay")["data"]
    train_loader, val_loader, test_loader = data_load_prepare(raw_data, ntx=NTX, nrx=NRX)

    # 2. 初始化模型
    model = DANN_SEINet(num_tx=NTX, num_rx=NRX).to(DEVICE)

    # 3. 训练
    print("\n--- Starting DANN Training ---")
    model = train_dann(model, train_loader, val_loader, epochs=50)

    # 4. 测试
    print("\n--- Final Testing ---")
    model.load_state_dict(torch.load("./model/dann_model_best.pth"))
    test_acc = test_dann(model, test_loader)
    print(f"Final Test TX Accuracy: {test_acc:.2f}%")