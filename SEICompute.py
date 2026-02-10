import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import math

def compute_prototypes(embeddings, labels, num_classes):
    """
    embeddings: (N, D)
    labels: (N,)
    """
    prototypes = []
    for c in range(num_classes):
        proto = embeddings[labels == c].mean(dim=0)
        proto = F.normalize(proto, dim=0)
        prototypes.append(proto)
    return torch.stack(prototypes)  # (C, D)

def cosine_proto_logits(query_embeddings, prototypes):
    """
    query_embeddings: (Q, D)
    prototypes: (C, D)
    """
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    logits = torch.matmul(query_embeddings, prototypes.t())  # cosine similarity
    return logits

def data_train_loader_prototype(data_cross_inputs, data_cross_labels, few_shot_num=10, batch_size=2048, ntx = 10, nrx = 5, samples = 800):
    """
    data_cross_inputs: (N, 3, 32, 32)
    data_cross_labels: (N, 2)  [tx, rx]
    """

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    tx_labels = data_cross_labels[:, 0]

    samples_per_class = samples * nrx

    for tx in range(ntx):
        # 取出该 tx 的所有样本索引
        idx = (tx_labels == tx).nonzero(as_tuple=True)[0]

        # 确保顺序一致（你的数据本来就是顺序的）
        idx = idx[:samples_per_class]

        # ---- 划分 ----
        train_idx = idx[:few_shot_num]

        remain = idx[few_shot_num:]
        half = remain.shape[0] // 2

        val_idx = remain[:half]
        test_idx = remain[half:]

        # ---- 收集 ----
        X_train.append(data_cross_inputs[train_idx])
        y_train.append(data_cross_labels[train_idx])

        X_val.append(data_cross_inputs[val_idx])
        y_val.append(data_cross_labels[val_idx])

        X_test.append(data_cross_inputs[test_idx])
        y_test.append(data_cross_labels[test_idx])

    # ---- 拼接 ----
    X_train = torch.cat(X_train, dim=0)
    y_train = torch.cat(y_train, dim=0)

    X_val = torch.cat(X_val, dim=0)
    y_val = torch.cat(y_val, dim=0)

    X_test = torch.cat(X_test, dim=0)
    y_test = torch.cat(y_test, dim=0)
    
    # ---- DataLoader ----
    train_cross_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    val_cross_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=65536,
        pin_memory=True
    )
    test_cross_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=65536,
        pin_memory=True
    )
    return train_cross_loader, val_cross_loader, test_cross_loader
    
class LossPrototypes(nn.Module):
    def __init__(self, margin: int = 4, scale: float = 1.5):
        super(LossPrototypes, self).__init__()
        self.margin = margin
        self.scale = scale
        
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, cosine: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = torch.clamp(cosine, -1 + 1e-7, 1 - 1e-7)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, labels)
        return loss
    
