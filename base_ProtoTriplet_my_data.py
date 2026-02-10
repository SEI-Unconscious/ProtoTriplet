import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from data_utilities import load_compact_pkl_dataset
from scipy.signal import stft
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
from SEINet import SEINet_Base_Prototype
from SEILoss import LossPrototypes

# import random
# seed = 43
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# ==================== Loss ====================
class ProtoTriplet(nn.Module):
    """
    Margin-based Triplet Loss with learnable prototypes
    """
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, features, prototypes, labels):
        B, _ = features.size()
        features = F.normalize(features, dim=1)
        prototypes = F.normalize(prototypes, dim=1)
        loss = 0.0
        for i in range(B):
            anchor = features[i]
            pos_proto = prototypes[labels[i]]
            neg_proto = torch.cat([prototypes[:labels[i]], prototypes[labels[i]+1:]], dim=0)
            d_pos = F.pairwise_distance(anchor.unsqueeze(0), pos_proto.unsqueeze(0))
            d_neg = F.pairwise_distance(anchor.unsqueeze(0), neg_proto)
            triplet_loss = F.relu(d_pos - d_neg + self.margin)
            loss += triplet_loss.mean()
        loss = loss / B
        return loss

# ==================== 数据处理 ====================
ntx = 7
nrx = 3

def iq_fft(iq_data: np.ndarray, n_fft = 32, window = "hann", fs = 1.0):
    assert iq_data.shape == (256, 2)
    x = iq_data[:, 0] + 1j * iq_data[:, 1]
    _, _, Zxx = stft(
        x, fs=fs, window=window, nperseg=n_fft,
        return_onesided=False, boundary=None, padded=False                  # type: ignore
    )
    return Zxx, np.abs(Zxx)

def pad_time_axis(Zxx, target_time = 32):
    freq, time = Zxx.shape
    assert freq == 32
    pad_width = target_time - time
    Z_pad = np.pad(Zxx, pad_width=((0,0),(0,pad_width)), mode='constant', constant_values=0)
    return Z_pad, np.abs(Z_pad)

def sftf_to_3_channels(Zxx_32x32):
    x = torch.from_numpy(Zxx_32x32).float().unsqueeze(0)
    x = x.repeat(3,1,1).unsqueeze(0)
    return x

def data_load_prepare(ntx=7, nrx=3, batch_size=64, file_path="../myDatabase/myDatabase.npy"):
    data = np.load(file_path, allow_pickle=True)
    data_inputs = []
    data_labels = []
    for txi in range(ntx):
        for rxi in range(nrx):
            for num in range(400):
                iq_data = data[txi][rxi][num]  # (256)
                iq_data_real = np.real(iq_data)
                iq_data_imag = np.imag(iq_data)
                iq_data_combined = np.stack((iq_data_real, iq_data_imag), axis=-1)  # (256, 2)
                _, abs_zxx = iq_fft(iq_data_combined)
                _, z_abs_32x32 = pad_time_axis(abs_zxx, target_time=32)
                tensor_data = sftf_to_3_channels(z_abs_32x32)
                data_inputs.append(tensor_data)
                data_labels.append([txi, rxi])
    data_inputs = torch.cat(data_inputs, dim=0)
    data_labels = torch.tensor(data_labels)
    X_train, X_temp, y_train, y_temp = train_test_split(data_inputs, data_labels, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)
    return train_loader, val_loader, test_loader

# ==================== 训练与测试 ====================
def train_model(model, train_loader, val_loader, num_epoch=50, lambda_triplet=0.1, lr=1e-3, save_path="./model/base_proto_triplet_my_data.pth"):
    device = next(model.parameters()).device
    proto_criterion = LossPrototypes()
    triplet_criterion = ProtoTriplet(margin=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")

    for epoch in range(num_epoch):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            tx_labels = labels[:,0]

            optimizer.zero_grad()
            features, cosine = model(inputs)
            loss_proto = proto_criterion(cosine, tx_labels)
            loss_triplet = triplet_criterion(features, model.prototype_layer.weight, tx_labels)
            loss = loss_proto + lambda_triplet * loss_triplet
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)

        train_loss = total_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                tx_labels = labels[:,0]
                features, cosine = model(inputs)
                loss_proto = proto_criterion(cosine, tx_labels)
                loss_triplet = triplet_criterion(features, model.prototype_layer.weight, tx_labels)
                val_loss += (loss_proto + lambda_triplet*loss_triplet).item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"[Epoch {epoch+1}/{num_epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✔ Model saved (val loss = {best_val_loss:.4f})")

def test_model(model, test_loader):
    device = next(model.parameters()).device
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            tx_labels = labels[:,0]
            _, cosine = model(inputs)
            preds = torch.argmax(cosine, dim=1)
            correct += (preds == tx_labels).sum().item()
            total += tx_labels.size(0)
    acc = 100.0 * correct / total
    print(f"Test Accuracy (TX): {acc:.2f}%")
    return acc

# ==================== 主程序 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lambda_triplet", type=float, help="weight of ProtoTriplet loss")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = data_load_prepare(ntx=ntx, nrx=nrx, batch_size=4096, file_path = "../myDatabase/myDatabase.npy")

    model = SEINet_Base_Prototype(num_classes=ntx).to(device)
    train_model(model, train_loader, val_loader, num_epoch=50, lambda_triplet=args.lambda_triplet)
    test_model(model, test_loader)
