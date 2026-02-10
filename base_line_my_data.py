from data_utilities import load_compact_pkl_dataset
import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from SEINet import SEINet_Baseline
from datetime import datetime
import pandas

import random
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# global variables
ntx = 7
nrx = 3
few_shot_nums = [1, 2, 3, 5, 10, 20]
current_date = str(datetime.now().year) + '_' + str(datetime.now().month).zfill(2) + '_' + str(datetime.now().day).zfill(2)
print(f"Today is {current_date}")

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

def data_load_prepare(ntx = 7, nrx = 3, file_path = "myDatabase/myDatabase.npy"):
    data = np.load(file_path, allow_pickle=True)
    data_cross_inputs = []
    data_cross_labels = []
    rxi = nrx
    for txi in range(ntx):
        for num in range(400):
            iq_data = data[txi][rxi][num]  # (256)
            iq_data_real = np.real(iq_data)
            iq_data_imag = np.imag(iq_data)
            iq_data_combined = np.stack((iq_data_real, iq_data_imag), axis=-1)  # (256, 2)
            _, abs_zxx = iq_fft(iq_data_combined)  # Zxx: (32, 15), abs_zxx: (32, 15)
            _, z_abs_32x32 = pad_time_axis(abs_zxx, target_time=32)  # z_pad: (32, 32)
            tensor_data = sftf_to_3_channels(z_abs_32x32)  # (1, 3, 32, 32)
            data_cross_inputs.append(tensor_data)
            data_cross_labels.append([txi, 0])

    data_cross_inputs = torch.cat(data_cross_inputs, dim=0)  # (ntx*nrx*num, 3, 32, 32) = (40000, 3, 32, 32)
    data_cross_labels = torch.tensor(data_cross_labels)  # (ntx*nrx, 2)
    
    return data_cross_inputs, data_cross_labels
        

def data_train_loader(data_cross_inputs, data_cross_labels, few_shot_num = 10, batch_size = 2048):
    train_cross_loader = None
    val_cross_loader = None

    if few_shot_num:
        X_cross_train, X_cross_temp, y_cross_train, y_cross_temp = train_test_split(
            data_cross_inputs, data_cross_labels,
            test_size=(1 - few_shot_num / 400),
            random_state=42
        )
        X_cross_val, X_cross_test, y_cross_val, y_cross_test = train_test_split(
            X_cross_temp, y_cross_temp,
            test_size=0.5,
            random_state=42
        )

        train_cross_loader = DataLoader(
            TensorDataset(X_cross_train, y_cross_train),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )

        val_cross_loader = DataLoader(
            TensorDataset(X_cross_val, y_cross_val),
            batch_size=batch_size,
            pin_memory=True
        )

    else:
        X_cross_test = data_cross_inputs
        y_cross_test = data_cross_labels

    test_cross_loader = DataLoader(
        TensorDataset(X_cross_test, y_cross_test),
        batch_size=batch_size,
        pin_memory=True
    )

    return train_cross_loader, val_cross_loader, test_cross_loader

def cross_model_train(model, train_cross_loader, val_cross_loader, num_epoch = 50, model_save_path = "./model/", time = "2025_12_23", index = 1):
    if train_cross_loader == None or val_cross_loader == None:
        return model
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_loss = float('inf')

    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_cross_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            _, outputs = model(inputs)
            loss = criterion(outputs, labels[:, 0])  # 只使用txi进行分类
            loss.backward()
            optimizer.step()
            running_loss += loss.detach() * inputs.size(0)

        epoch_loss = running_loss.item() / len(train_cross_loader.dataset) # type: ignore

        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_cross_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                _, outputs = model(inputs)
                loss = criterion(outputs, labels[:, 0])  # 只使用txi进行分类
                val_running_loss += loss.detach() * inputs.size(0)

        val_epoch_loss = val_running_loss.item() / len(val_cross_loader.dataset)        #type:ignore

        # 保存最佳模型
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            torch.save(model.state_dict(), f"{model_save_path}best_model_baseline_my_data_{time}_{index}.pth")
            print(f'Model saved at epoch {epoch+1} with training loss {epoch_loss:.4f} and validation loss {best_loss:.4f}')

    return model

def cross_model_test(model, test_cross_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_cross_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            _, outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels[:, 0]).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def main():
    if device.type == 'cpu':
        raise NotImplementedError("CPU mode is not implemented in this version.")
    cross_test_accuracys = []
    print("===========================")
    print("Few-Shot Learning Start")
    print("===========================")
    data_cross_inputs, data_cross_labels = data_load_prepare(ntx=ntx, nrx=nrx, file_path="../myDatabase/myDatabase.npy")
    print("Data Preparation Completed")
    for few_shot_num in few_shot_nums:
        print("===========================")
        print(f"Running few-shot num: {few_shot_num} on 'gpu'")
        train_cross_loader, val_cross_loader, test_cross_loader = data_train_loader(data_cross_inputs, data_cross_labels, few_shot_num=few_shot_num, batch_size=128)
        print("===========================")
        print("Starting Cross Model Training")
        cross_model = SEINet_Baseline(num_classes=ntx, base_model_path="base_model_my_baseline.pth").to(device)
        cross_model = cross_model_train(model=cross_model, train_cross_loader=train_cross_loader, val_cross_loader=val_cross_loader, num_epoch=100, model_save_path="./model/", time=current_date, index=2)
        print("===========================")
        print("Starting Cross Model Testing")
        cross_test_accuracy = cross_model_test(model=cross_model, test_cross_loader=test_cross_loader)
        print(f"Cross Test Accuracy: {cross_test_accuracy:.4f}%")
        cross_test_accuracys.append(cross_test_accuracy)
        print("===========================")

    df = pandas.DataFrame({
        'few_shot_nums': few_shot_nums,
        'cross_test_accuracys': cross_test_accuracys
    })
    df.to_csv('../results/base_line_my_data' + '.csv', index=False)

if __name__ == '__main__':
    main()
