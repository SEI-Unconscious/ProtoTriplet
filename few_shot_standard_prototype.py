import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from datetime import datetime
import pandas as pd

from data_load import data_load, data_load_prepare, data_train_loader
from SEINet import SEINet_Base_Prototype

# ===============================
# 标准不可学习原型网络 (可训练 layer4)
# ===============================
class SEINet_FewShot_StandardProto(nn.Module):
    """
    Standard Prototype Network for few-shot learning
    - Backbone frozen except layer4
    - Prototypes computed from entire dataset features (non-trainable)
    """
    def __init__(self, base_model_path, num_classes):
        super().__init__()
        base_model = SEINet_Base_Prototype(num_classes=num_classes)
        base_model.load_state_dict(torch.load(base_model_path))
        self.backbone = base_model.backbone
        self.num_classes = num_classes
        self.prototypes = None  # will be computed from full dataset

        # Freeze backbone except layer4
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
            if "layer4" in name:
                param.requires_grad = True

    def compute_prototypes(self, features, labels):
        """
        Compute class prototypes from features
        features: [N, D]
        labels: [N]
        """
        prototypes = []
        for c in range(self.num_classes):
            class_feat = features[labels == c]
            if class_feat.size(0) == 0:
                prototypes.append(torch.zeros(features.size(1), device=features.device))
            else:
                prototypes.append(class_feat.mean(dim=0))
        self.prototypes = torch.stack(prototypes)  # [C, D]
        self.prototypes = F.normalize(self.prototypes, dim=1)

    def forward(self, x):
        feat = self.backbone(x)
        feat_norm = F.normalize(feat, dim=1)
        if self.prototypes is None:
            raise ValueError("Prototypes not computed. Call compute_prototypes() first.")
        cosine = torch.matmul(feat_norm, self.prototypes.t())  # [B, C]
        return cosine


# ===============================
# Proto Loss
# ===============================
class ProtoLoss(nn.Module):
    """
    Prototypical loss using cosine similarity + cross-entropy
    """
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, cosine, labels):
        return self.ce(cosine, labels)


# ===============================
# 特征提取函数
# ===============================
def extract_features_from_loader(model, data_loader, device):
    """
    Extract features from all data in data_loader
    - Returns: features [N, D], labels [N]
    - Handles None or empty loader
    """
    if data_loader is None or len(data_loader.dataset) == 0:
        return None, None

    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels[:, 0].to(device)
            feat = model.backbone(inputs)
            features_list.append(feat)
            labels_list.append(labels)

    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return features, labels


def extract_features_from_dataset(model, inputs, labels, device, batch_size=1024):
    """
    Extract features from full dataset tensors (inputs, labels)
    """
    dataset = torch.utils.data.TensorDataset(inputs, labels)                                    # type: ignore
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)         # type: ignore
    return extract_features_from_loader(model, loader, device)


# ===============================
# Few-shot Training
# ===============================
def fewshot_train(
    model,
    train_loader,
    val_loader,
    full_inputs,
    full_labels,
    num_epochs=50,
    lr=1e-3,
    save_path="./model/fewshot_standard_prototype.pth",
    device=torch.device("cuda")
):
    model.to(device)
    criterion = ProtoLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # -------- 先用全数据集计算原型 --------
    full_features, full_labels = extract_features_from_dataset(model, full_inputs, full_labels, device)
    model.compute_prototypes(full_features, full_labels)

    # few-shot=0 或训练集为空时直接返回
    if train_loader is None or len(train_loader.dataset) == 0:
        print("⚠️ No training samples, skip training.")
        return model

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        train_loss_total = 0.0

        for inputs_batch, labels_batch in train_loader:
            inputs_batch = inputs_batch.to(device)
            labels_batch = labels_batch[:, 0].to(device)

            optimizer.zero_grad()
            cosine = model(inputs_batch)
            loss = criterion(cosine, labels_batch)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * inputs_batch.size(0)

        train_loss_total /= len(train_loader.dataset)

        # 验证
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for inputs_val, labels_val in val_loader:
                inputs_val = inputs_val.to(device)
                labels_val = labels_val[:, 0].to(device)
                cosine = model(inputs_val)
                loss = criterion(cosine, labels_val)
                val_loss_total += loss.item() * inputs_val.size(0)
        val_loss_total /= len(val_loader.dataset)

        print(
            f"[Epoch {epoch+1:03d}] "
            f"Train Loss: {train_loss_total:.4f} | "
            f"Val Loss: {val_loss_total:.4f}"
        )

        if val_loss_total < best_val_loss:
            best_val_loss = val_loss_total
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved (val loss = {best_val_loss:.4f})")

    return model


# ===============================
# Few-shot Test
# ===============================
def fewshot_test(model, test_loader, device=torch.device("cuda")):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ntx = 7
    nrx = 5
    few_shot_nums = [1, 2, 3, 5, 10, 20, 50]

    current_date = datetime.now().strftime("%Y_%m_%d")
    print(f"Date: {current_date}, Device: {device}")

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

        model = SEINet_FewShot_StandardProto(
            base_model_path="./model/base_proto_supcon.pth",
            num_classes=ntx
        ).to(device)

        model = fewshot_train(
            model,
            train_loader,
            val_loader,
            full_inputs=inputs,
            full_labels=labels,
            num_epochs=50,
            lr=1e-3,
            save_path=f"./model/fewshot_standard_prototype_{few_shot}.pth",
            device=device
        )

        acc = fewshot_test(model, test_loader, device=device)
        print(f"🎯 Few-shot {few_shot} Test Accuracy: {acc:.2f}%")
        cross_test_accuracys.append(acc)

    # Save results
    df = pd.DataFrame({
        "few_shot_nums": few_shot_nums,
        "cross_test_accuracys": cross_test_accuracys
    })
    df.to_csv(f"../results/fewshot_standard_proto_{current_date}.csv", index=False)


if __name__ == "__main__":
    main()
