from data_utilities import load_compact_pkl_dataset
import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

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
    assert iq_batch.ndim == 3  # (N, 256, 2)
    _, abs_zxx_batch = iq_fft(iq_batch)  # (N, 32, 15)
    _, z_abs_32x32_batch = pad_time_axis(abs_zxx_batch, target_time=32)  # (N, 32, 32)
    tensor_data = sftf_to_3_channels(z_abs_32x32_batch)  # (N, 3, 32, 32)
    return tensor_data

def data_load_prepare(data, ntx = 10, nrx = 5, same_rx_flag = False):
    data_cross_inputs = []
    data_cross_labels = []
    for txi in range(ntx):
        if same_rx_flag:
            rxi = nrx
            for num in range(800):
                iq_data = data[txi][rxi][0][1][num]  # (256, 2)
                _, abs_zxx = iq_fft(iq_data)  # Zxx: (32, 15), abs_zxx: (32, 15)
                _, z_abs_32x32 = pad_time_axis(abs_zxx, target_time=32)  # z_pad: (32, 32)
                tensor_data = sftf_to_3_channels(z_abs_32x32)  # (1, 3, 32, 32)
                data_cross_inputs.append(tensor_data)
                data_cross_labels.append([txi, rxi])
        else:
            for rxi in range(nrx, nrx * 2):
                for num in range(800):
                    iq_data = data[txi][rxi][0][1][num]  # (256, 2)
                    _, abs_zxx = iq_fft(iq_data)  # Zxx: (32, 15), abs_zxx: (32, 15)
                    _, z_abs_32x32 = pad_time_axis(abs_zxx, target_time=32)  # z_pad: (32, 32)
                    tensor_data = sftf_to_3_channels(z_abs_32x32)  # (1, 3, 32, 32)
                    data_cross_inputs.append(tensor_data)
                    data_cross_labels.append([txi, rxi - nrx])

    data_cross_inputs = torch.cat(data_cross_inputs, dim=0)  # (ntx*nrx*num, 3, 32, 32) = (40000, 3, 32, 32)
    data_cross_labels = torch.tensor(data_cross_labels)  # (ntx*nrx, 2)
    
    return data_cross_inputs, data_cross_labels
        

def data_train_loader(data_cross_inputs, data_cross_labels, few_shot_num=10, batch_size=2048, samples=800):
    train_cross_loader = None
    val_cross_loader = None

    if few_shot_num:
        X_cross_train, X_cross_temp, y_cross_train, y_cross_temp = train_test_split(
            data_cross_inputs, data_cross_labels,
            test_size=(1 - few_shot_num / samples),
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
