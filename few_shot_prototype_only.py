import torch
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

    criterion = LossPrototypes()

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

            cosine = model(inputs)             # ⚠️ cosine similarity
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

                cosine = model(inputs)
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

            outputs = model(inputs)
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
    nrx = 5
    few_shot_nums = [0, 1, 2, 3, 5, 10, 20, 50]

    current_date = datetime.now().strftime("%Y_%m_%d")
    print(f"Date: {current_date}")
    print(f"Device: {device}")

    # Load dataset
    data = data_load()
    inputs, labels = data_load_prepare(data, ntx=ntx, nrx=nrx, same_rx_flag=True)

    cross_test_accuracys = []

    for few_shot in few_shot_nums:
        print("=" * 40)
        print(f"Few-shot = {few_shot}")

        train_loader, val_loader, test_loader = data_train_loader(
            inputs,
            labels,
            few_shot_num=few_shot,
            batch_size=2048,
            samples=800
        )

        model = SEINet_FewShot_Mapping(
            base_model_path="./model/base_proto_only.pth",
            num_classes=ntx,
            mapping=False
        ).to(device)

        model = fewshot_train(
            model,
            train_loader,
            val_loader,
            num_epochs=100,
            lr=1e-3,
            save_path=f"./model/fewshot_best_only_prototype_{few_shot}.pth"
        )

        acc = fewshot_test(model, test_loader)
        print(f"🎯 Few-shot {few_shot} Test Accuracy: {acc:.2f}%")

        cross_test_accuracys.append(acc)

    # Save results
    df = pd.DataFrame({
        "few_shot_nums": few_shot_nums,
        "cross_test_accuracys": cross_test_accuracys
    })
    df.to_csv(f"../results/fewshot_only_prototype_{current_date}.csv", index=False)


if __name__ == "__main__":
    main()
