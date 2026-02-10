from data_utilities import load_compact_pkl_dataset
import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from SEINet import SEINet_Base
import random

seed = 46
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# global variables
ntx = 7
nrx = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def plot_show(Zxx, title = "STFT Magnitude"):
    mag = np.abs(Zxx)
    plt.figure()
    plt.imshow(
        mag,
        origin = "lower",
        aspect = "auto"
        )
    plt.colorbar(label = "Magnitude")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    plt.show()

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

def data_load_prepare(ntx = 7, nrx = 3, batch_size = 64, file_path = "myDatabase/myDatabase.npy"):
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
                _, abs_zxx = iq_fft(iq_data_combined)  # Zxx: (32, 15), abs_zxx: (32, 15)
                _, z_abs_32x32 = pad_time_axis(abs_zxx, target_time=32)  # z_pad: (32, 32)
                tensor_data = sftf_to_3_channels(z_abs_32x32)  # (1, 3, 32, 32)
                data_inputs.append(tensor_data)
                data_labels.append([txi, rxi])

    data_inputs = torch.cat(data_inputs, dim=0)  # (ntx*nrx*num, 3, 32, 32) = (8400, 3, 32, 32)
    data_labels = torch.tensor(data_labels)  # (ntx*nrx, 2)
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


def base_model_train(model, train_loader, val_loader, num_epoch = 50):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_loss = float('inf')

    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            _, outputs = model(inputs)
            loss = criterion(outputs, labels[:,0])  # Assuming we are classifying based on the first label (tx index)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                _, outputs = model(inputs)
                loss = criterion(outputs, labels[:,0])
                val_loss += loss.item() * inputs.size(0)

        val_epoch_loss = val_loss / len(val_loader.dataset)

        print(f'Epoch {epoch+1}/{num_epoch}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_epoch_loss:.4f}')

        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            torch.save(model.state_dict(), f"./model/base_model_my_baseline.pth")
            print(f'Model saved at epoch {epoch+1} with validation loss {best_loss:.4f}')

    return model

def base_model_test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels[:, 0]).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def main():
    train_loader, val_loader, test_loader = data_load_prepare(ntx=ntx, nrx=nrx, batch_size=1024, file_path = "../myDatabase/myDatabase.npy")
    print("===========================")
    print("Data Preparation Completed")
    print("===========================")
    base_model = SEINet_Base(num_classes=ntx).to(device)
    print("===========================")
    print("Starting Base Model Training")
    base_model = base_model_train(model=base_model, train_loader=train_loader, val_loader=val_loader, num_epoch=50)
    print("===========================")
    print("Starting Base Model Testing")
    test_accuracy = base_model_test(model=base_model, test_loader=test_loader)
    print(f"Test Accuracy: {test_accuracy:.4f}%")
    print("===========================")

if __name__ == '__main__':
    main()


 