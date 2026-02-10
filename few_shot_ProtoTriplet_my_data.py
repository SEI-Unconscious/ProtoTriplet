import torch
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.signal import stft
from SEINet import SEINet_FewShot_Mapping
from SEILoss import LossPrototypes
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# import random
# seed = 43
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# ===============================
# 全局配置
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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


# ===============================
# Main
# ===============================
def main():
    ntx = 7
    nrx = 3
    few_shot_nums = [1, 2, 3, 5, 10, 20]

    current_date = datetime.now().strftime("%Y_%m_%d")
    print(f"Date: {current_date}")
    print(f"Device: {device}")

    # Load dataset
    inputs, labels = data_load_prepare(ntx=ntx, nrx=nrx, file_path="../myDatabase/myDatabase.npy")

    cross_test_accuracys = []

    for few_shot in few_shot_nums:
        print("=" * 40)
        print(f"Few-shot = {few_shot}")

        train_loader, val_loader, test_loader = data_train_loader(
            inputs,
            labels,
            few_shot_num=few_shot,
            batch_size=2048
        )

        model = SEINet_FewShot_Mapping(
            base_model_path="./model/base_proto_triplet_my_data.pth",
            num_classes=ntx,
            mapping=False
        ).to(device)

        model = fewshot_train(
            model,
            train_loader,
            val_loader,
            num_epochs=100,
            lr=1e-3,
            save_path=f"./model/fewshot_triplet_my_data_{few_shot}.pth"
        )

        acc = fewshot_test(model, test_loader)
        print(f"🎯 Few-shot {few_shot} Test Accuracy: {acc:.2f}%")

        cross_test_accuracys.append(acc)

    # Save results
    df = pd.DataFrame({
        "few_shot_nums": few_shot_nums,
        "cross_test_accuracys": cross_test_accuracys
    })
    df.to_csv("../results/fewshot_Triplet_My_Data.csv", index=False)


if __name__ == "__main__":
    main()
