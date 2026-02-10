import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math
from torch.utils.data import DataLoader, TensorDataset
from data_utilities import load_compact_pkl_dataset
import numpy as np
from scipy.signal import stft
from sklearn.model_selection import train_test_split
import argparse
from SEINet import SEINet_Base_Prototype
from SEILoss import SupConLoss, LossPrototypes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
ntx = 7
nrx = 5

def data_load(dataset_path = "../dataset/", dataset_name = "SingleDay"):
    dataset = load_compact_pkl_dataset(dataset_path, dataset_name)
    data = dataset["data"]
    print("DataSet Load Successfully")
    return data

def iq_fft(iq_data: np.ndarray, n_fft = 32, window = "hann", fs = 1.0):    #(17, 15, 2)
    assert iq_data.shape == (256, 2)

    x = iq_data[:, 0] + 1j * iq_data[:, 1]

    _, _, Zxx = stft(
        x,
        fs = fs,
        window = window,
        nperseg = n_fft,
        return_onesided = False,
        boundary = None, # type: ignore
        padded = False
    )

    return Zxx, np.abs(Zxx)  # Zxx: (32, 15), abs_zxx: (32, 15)

def pad_time_axis(Zxx, target_time = 32):
    freq, time = Zxx.shape
    assert freq == 32
    assert time <= target_time
    pad_width = target_time - time
    Z_pad = np.pad(
        Zxx,
        pad_width=((0, 0), (0, pad_width)),  # only for time axis
        mode='constant',
        constant_values=0
    )
    return Z_pad, np.abs(Z_pad)  # z_pad: (32, 32)

def sftf_to_3_channels(Zxx_32x32):      # We can also use a convolution layer to convert 1 channel to 3 channels
    x = torch.from_numpy(Zxx_32x32).float().unsqueeze(0)
    x = x.repeat(3, 1, 1)
    x = x.unsqueeze(0)  # (1, 3, 32, 32)
    return x

# 批处理
def batch_process_iq(iq_batch):
    _, abs_zxx_batch = iq_fft(iq_batch)  # (N, 32, 15)
    _, z_abs_32x32_batch = pad_time_axis(abs_zxx_batch, target_time=32)  # (N, 32, 32)
    tensor_data = sftf_to_3_channels(z_abs_32x32_batch)  # (N, 3, 32, 32)
    return tensor_data

def data_load_prepare(data, ntx = 10, nrx = 5, batch_size = 64):
    data_inputs = []
    data_labels = []
    for txi in range(ntx):
        for rxi in range(nrx):
            for num in range(800):
                iq_data = data[txi][rxi][0][1][num]  # (256, 2)
                _, abs_zxx = iq_fft(iq_data)  # Zxx: (32, 15), abs_zxx: (32, 15)
                _, z_abs_32x32 = pad_time_axis(abs_zxx, target_time=32)  # z_pad: (32, 32)
                tensor_data = sftf_to_3_channels(z_abs_32x32)  # (1, 3, 32, 32)
                data_inputs.append(tensor_data)
                data_labels.append([txi, rxi])

    data_inputs = torch.cat(data_inputs, dim=0)  # (ntx*nrx*num, 3, 32, 32) = (40000, 3, 32, 32)
    data_labels = torch.tensor(data_labels)  # (ntx*nrx*num, 2) = (40000, 2)
    X_train, X_temp, y_train, y_temp = train_test_split(
        data_inputs, data_labels, test_size=0.2, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    # 创建DataLoader
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size
    )
    return train_loader, val_loader, test_loader

def base_model_train(
    model,
    train_loader,
    val_loader,
    num_epoch=50,
    lambda_supcon=0.1,
    lr=1e-3,
    save_path="./model/base_proto_supcon.pth"
):
    device = next(model.parameters()).device

    proto_criterion = LossPrototypes()
    supcon_criterion = SupConLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(num_epoch):
        model.train()
        total_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            tx_labels = labels[:, 0]   # 只用 TX

            optimizer.zero_grad()

            features, cosine = model(inputs)

            loss_proto = proto_criterion(cosine, tx_labels)
            loss_supcon = supcon_criterion(features, tx_labels)

            loss = loss_proto + lambda_supcon * loss_supcon
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)

        train_loss = total_loss / len(train_loader.dataset)

        # ===== Validation =====
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                tx_labels = labels[:, 0]

                features, cosine = model(inputs)
                loss_proto = proto_criterion(cosine, tx_labels)
                loss_supcon = supcon_criterion(features, tx_labels)
                loss = loss_proto + lambda_supcon * loss_supcon

                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)

        print(
            f"[Epoch {epoch+1}/{num_epoch}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✔ Model saved (val loss = {best_val_loss:.4f})")

def base_model_test(model, test_loader):
    device = next(model.parameters()).device
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            tx_labels = labels[:, 0]  # 只评估 TX 分类

            _, cosine = model(inputs)
            preds = torch.argmax(cosine, dim=1)

            correct += (preds == tx_labels).sum().item()
            total += tx_labels.size(0)

    acc = 100.0 * correct / total
    print(f"Base Model Test Accuracy (TX): {acc:.2f}%")

    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lambda_c", type=float, help="weight of supervised contrastive loss")
    args = parser.parse_args()
    data = data_load()
    train_loader, val_loader, test_loader = data_load_prepare(
        data,
        ntx=ntx,
        nrx=nrx,
        batch_size=4096
    )
    model = SEINet_Base_Prototype(num_classes=ntx).to(device)
    base_model_train(
        model,
        train_loader,
        val_loader,
        num_epoch=50,
        lambda_supcon=args.lambda_c,
        save_path="./model/base_proto_supcon.pth"
    )

    base_model_test(model, test_loader)

