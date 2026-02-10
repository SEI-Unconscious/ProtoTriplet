import torch
import torch.nn as nn
import math

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        B = features.size(0)

        features = nn.functional.normalize(features, dim=1)
        labels = labels.view(-1, 1)

        mask = torch.eq(labels, labels.T).float().to(device)

        sim = torch.matmul(features, features.T) / self.temperature

        logits_mask = torch.ones_like(mask) - torch.eye(B, device=device)
        mask = mask * logits_mask

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
        loss = -mean_log_prob_pos.mean()
        return loss

class LossPrototypes(nn.Module):
    def __init__(self, margin=0.5, scale=16.0, mapping=False):
        super().__init__()
        self.mapping_flag = mapping
        self.margin = margin
        if not self.mapping_flag:
            self.scale = scale
        else:
            self.scale = nn.Parameter(torch.tensor(scale)) 

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        self.ce = nn.CrossEntropyLoss()

    def forward(self, cosine, labels):
        cosine = torch.clamp(cosine, -1 + 1e-7, 1 - 1e-7)
        sine = torch.sqrt(1.0 - cosine ** 2)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.scale

        return self.ce(logits, labels)
