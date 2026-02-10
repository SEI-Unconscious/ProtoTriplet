import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from datetime import datetime
import pandas as pd

from data_load import data_load, data_load_prepare, data_train_loader
from SEINet import SEINet_FewShot_Mapping
from SEILoss import LossPrototypes

# ===============================
# 全局配置
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===============================
# Few-shot Training
# ===============================
def fewshot_train(
    model,
    train_loader,
    val_loader,
    num_epochs=50,
    lr=1e-3,
    save_path="./model/fewshot_best.pth"
):
    if train_loader == None or val_loader == None:
        return model

    device = next(model.parameters()).device

    criterion = LossPrototypes(mapping=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # ============ Train ============
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels[:, 0].to(device)   # TX label

            optimizer.zero_grad()

            _, cosine = model(inputs)             # ⚠️ cosine similarity
            loss = criterion(cosine, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        # ============ Validation ============
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels[:, 0].to(device)

                _, cosine = model(inputs)
                loss = criterion(cosine, labels)

                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)

        print(
            f"[Epoch {epoch+1:03d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved (val loss = {best_val_loss:.4f})")

    return model

# ===============================
# Few-shot Test
# ===============================
def fewshot_test(model, test_loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels[:, 0].to(device)

            _, outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = 100.0 * correct / total
    return acc


import numpy as np

def add_awgn_to_iq(iq_data, snr_db):
    """
    对 (256, 2) 的 I/Q 信号添加高斯白噪声
    """
    # 分离 I 和 Q
    i = iq_data[:, 0]
    q = iq_data[:, 1]
    
    # 计算信号功率 P = mean(I^2 + Q^2)
    sig_power = np.mean(i**2 + q**2)
    
    # SNR 转线性增益
    snr_linear = 10**(snr_db / 10.0)
    noise_power = sig_power / snr_linear
    
    # 生成噪声 (均值为0，方差为 noise_power/2 每路)
    # 这样 I+Q 总的噪声功率就是 noise_power
    noise_std = np.sqrt(noise_power / 2)
    noise_i = np.random.normal(0, noise_std, i.shape)
    noise_q = np.random.normal(0, noise_std, q.shape)
    
    # 合并返回
    return np.stack([i + noise_i, q + noise_q], axis=1)

from data_load import iq_fft, pad_time_axis, sftf_to_3_channels

def data_load_prepare_with_noise(data, ntx=10, nrx=5, same_rx_flag=False, snr_db=None):
    data_cross_inputs = []
    data_cross_labels = []
    
    for txi in range(ntx):
        # 确定接收机索引范围
        rx_range = [nrx] if same_rx_flag else range(nrx, nrx * 2)
        
        for rxi in rx_range:
            for num in range(800):
                # 1. 提取原始 I/Q 数据 (256, 2)
                iq_data = data[txi][rxi][0][1][num]
                
                # 2. 如果指定了 SNR，则在此处加噪
                if snr_db is not None:
                    iq_data = add_awgn_to_iq(iq_data, snr_db)
                
                # 3. 后续变换过程保持不变
                _, abs_zxx = iq_fft(iq_data)  
                _, z_abs_32x32 = pad_time_axis(abs_zxx, target_time=32)
                tensor_data = sftf_to_3_channels(z_abs_32x32)
                
                data_cross_inputs.append(tensor_data)
                label_rxi = rxi if same_rx_flag else rxi - nrx
                data_cross_labels.append([txi, label_rxi])

    data_cross_inputs = torch.cat(data_cross_inputs, dim=0)
    data_cross_labels = torch.tensor(data_cross_labels)
    
    return data_cross_inputs, data_cross_labels

# ===============================
# Main
# ===============================
def main():
    ntx = 7
    nrx = 5
    snr_list = [0, 5, 10, 15, 20, 25, 30]
    # 你要求的具体 Shot 列表
    few_shot_nums = [1, 2, 3, 5, 10] 

    current_date = datetime.now().strftime("%Y_%m_%d")
    data = data_load() # 原始数据只加载一次

    results_records = []

    for snr in snr_list:
        print(f"\n{'='*20} Current SNR: {snr}dB {'='*20}")
        
        # --- 关键步骤：在当前 SNR 下重新生成 STFT 特征 ---
        inputs, labels = data_load_prepare_with_noise(
            data, ntx=ntx, nrx=nrx, same_rx_flag=True, snr_db=snr
        )
        
        current_snr_accs = {"SNR": snr}

        for shot in few_shot_nums:
            print(f"\n>>> SNR {snr}dB | Testing {shot}-shot")

            train_loader, val_loader, test_loader = data_train_loader(
                inputs, labels, few_shot_num=shot, batch_size=2048, samples=800
            )

            # 重新初始化模型，避免权重累积
            model = SEINet_FewShot_Mapping(
                base_model_path="./model/base_proto_triplet.pth",
                num_classes=ntx,
                mapping=False
            ).to(device)

            # 训练 (可以适当减小 epoch 以加快速度，如 50)
            model = fewshot_train(
                model, train_loader, val_loader, 
                num_epochs=100, lr=1e-3,
                save_path=f"./model/best_{snr}dB_{shot}shot.pth"
            )

            acc = fewshot_test(model, test_loader)
            current_snr_accs[f"{shot}-shot"] = acc                  # type: ignore
            print(f"Result: {acc:.2f}%")

        results_records.append(current_snr_accs)

    # 保存最终表格
    df = pd.DataFrame(results_records)
    # 调整列顺序，确保 SNR 在第一列
    cols = ["SNR"] + [f"{s}-shot" for s in few_shot_nums]
    df = df[cols]
    df.to_csv(f"../results/SNR_FewShot_STFT_{current_date}.csv", index=False)
    print(df)

if __name__ == "__main__":
    main()
