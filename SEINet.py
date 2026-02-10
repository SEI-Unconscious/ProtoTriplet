import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torchvision.models as models
import torch
import torch.nn.functional as F


class SEINet_Base(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.backbone = models.resnet18(weights=None)
        self.backbone.maxpool = nn.Identity()       # type: ignore
        self.backbone.fc = nn.Identity()            # type: ignore

        # self.classifier = nn.Linear(512, num_classes)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return features, logits
    
class SEINet_Baseline(nn.Module):
    def __init__(self, num_classes=10, base_model_path="base_model_resnet18_baseline.pth"):
        super().__init__()
        
        self.base_model_path = base_model_path
        self.base_model = SEINet_Base(num_classes=num_classes)
        self.backbone = self.base_model.backbone
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self.backbone_load(base_model_name=base_model_path)

        # -------- 冻结全部 --------
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return features, logits
    
    def backbone_load(self, base_model_path = './model/', base_model_name = "base_model_resnet18_baseline.pth"):
        self.base_model.load_state_dict(torch.load(base_model_path + base_model_name))

class SEINet_FewShotClassifier_MLP(nn.Module):
    def __init__(self, num_classes=10, base_model_name='base_model_resnet18.pth'):
        super(SEINet_FewShotClassifier_MLP, self).__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.maxpool = nn.Identity()       # type: ignore
        self.backbone.fc = nn.Identity()            # type: ignore
        # 加载 base model backbone
        self.backbone_load(base_model_name=base_model_name)

        # -------- 冻结全部 --------
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
    
    def backbone_load(self, base_model_path = './model/', base_model_name = 'base_model_resnet18.pth'):
        self.backbone.load_state_dict(torch.load(base_model_path + base_model_name))
    
class SEINet_FewShot_PartialUnfreeze(nn.Module):
    def __init__(
        self,
        num_classes=10,
        base_model_name='base_model_resnet18.pth',
        unfreeze_layer4=True,
        unfreeze_layer3=False
    ):
        super().__init__()

        # -------- Backbone --------
        self.backbone = models.resnet18(weights=None)
        self.backbone.maxpool = nn.Identity()       # type: ignore
        self.backbone.fc = nn.Identity()            # type: ignore

        # 加载 base model backbone
        self.backbone_load(base_model_name=base_model_name)

        # -------- 冻结全部 --------
        for p in self.backbone.parameters():
            p.requires_grad = False

        # -------- 部分解冻 --------
        if unfreeze_layer4:
            for p in self.backbone.layer4.parameters():
                p.requires_grad = True

        if unfreeze_layer3:
            for p in self.backbone.layer3.parameters():
                p.requires_grad = True

        # -------- Few-shot classifier --------
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def backbone_load(self, base_model_path = './model/', base_model_name = 'base_model_resnet18.pth'):
        self.backbone.load_state_dict(torch.load(base_model_path + base_model_name))

class SEINet_FewShot_ProtoNet_Train(nn.Module):
    def __init__(self, base_model_name="base_model_resnet18.pth"):
        super().__init__()

        # -------- Backbone --------
        self.backbone = models.resnet18(weights=None)
        self.backbone.maxpool = nn.Identity()   # type: ignore
        self.backbone.fc = nn.Identity()        # type: ignore

        # Load trained base backbone
        self.backbone_load(base_model_name=base_model_name)

        for p in self.backbone.parameters():
            p.requires_grad = False
        
        for p in self.backbone.layer4.parameters():
            p.requires_grad = True

        self.prototype_layer = nn.Linear(512, 10, bias=False)  # 10 classes for tx

    def forward(self, x):
        features = self.backbone(x)
        feature_worm = torch.nn.functional.normalize(features, p=2, dim=1)  
        w_norm = torch.nn.functional.normalize(self.prototype_layer.weight, p=2, dim=1)
        cosine_similarity = torch.mm(feature_worm, w_norm.t())
        return cosine_similarity
    
    def backbone_load(self, base_model_path="./model/", base_model_name="base_model_resnet18.pth"):
        self.backbone.load_state_dict(torch.load(base_model_path + base_model_name, map_location="cpu"))

    def get_prototype(self):
        return self.prototype_layer.weight.data

class SEINet_Base_Prototype(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.backbone = models.resnet18(weights=None)
        self.backbone.maxpool = nn.Identity()           # type: ignore
        self.backbone.fc = nn.Identity()                # type: ignore

        self.prototype_layer = nn.Linear(512, num_classes, bias=False)
        nn.init.xavier_uniform_(self.prototype_layer.weight)

    def forward(self, x):
        features = self.backbone(x)              # (B, 512)
        feat_norm = F.normalize(features, dim=1)
        proto_norm = F.normalize(self.prototype_layer.weight, dim=1)
        cosine = torch.matmul(feat_norm, proto_norm.t())  # (B, C)
        return features, cosine

class SEINet_FewShot_Mapping(nn.Module):
    """
    Base ProtoNet + lightweight mapping
    Only layer4 and mapping are trainable
    """
    def __init__(self, base_model_path, num_classes, mapping=True):
        super().__init__()

        # 1️⃣ load base model
        base_model = SEINet_Base_Prototype(num_classes=num_classes)
        base_model.load_state_dict(torch.load(base_model_path))

        self.backbone = base_model.backbone
        self.prototypes = base_model.prototype_layer
        self.mapping_flag = mapping
        # 2️⃣ mapping layer (identity init)
        if self.mapping_flag:
            self.mapping = nn.Linear(512, 512, bias=False)
            nn.init.eye_(self.mapping.weight)

        # 3️⃣ 冻结 backbone except layer4
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
            if "layer4" in name:
                param.requires_grad = True

        # prototypes & mapping trainable
        for p in self.prototypes.parameters():
            p.requires_grad = True
        if self.mapping_flag:
            for p in self.mapping.parameters():
                p.requires_grad = True
        

    def forward(self, x):
        feat = self.backbone(x)          # [B, 512]
        if self.mapping_flag:
            feat = self.mapping(feat)
        feat_norm = F.normalize(feat, dim=1)
        proto_norm = F.normalize(self.prototypes.weight, dim=1)
        cosine = torch.matmul(feat_norm, proto_norm.t())  # [B, C]
        return feat, cosine

from torch.autograd import Function
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播时：梯度取反并乘以 alpha
        output = grad_output.neg() * ctx.alpha
        return output, None

class DANN_SEINet(nn.Module):
    def __init__(self, num_tx=7, num_rx=5):
        super(DANN_SEINet, self).__init__()
        
        # 特征提取器 (Backbone)
        self.backbone = models.resnet18(weights=None)
        self.backbone.maxpool = nn.Identity()               # type: ignore
        self.backbone.fc = nn.Identity()                    # type: ignore

        # TX 分类器 (主任务)
        self.tx_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_tx)
        )
        
        # RX 判别器 (对抗任务)
        self.rx_discriminator = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_rx)
        )

    def forward(self, x, alpha=1.0):
        # 1. 提取特征
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # 2. TX 分类逻辑 (正常路径)
        tx_logits = self.tx_classifier(features)
        
        # 3. RX 分类逻辑 (对抗路径：通过 GRL)
        reverse_features = ReverseLayerF.apply(features, alpha)
        rx_logits = self.rx_discriminator(reverse_features)
        
        return tx_logits, rx_logits

class SEINet_DANN_FewShot(nn.Module):
    def __init__(self, num_classes=7, dann_model_path="./model/dann_model_best.pth"):
        super().__init__()
        
        # 1. 结构初始化 (必须与 DANN 训练时的 backbone 结构一致)
        self.backbone = models.resnet18(weights=None)
        self.backbone.maxpool = nn.Identity()               # type: ignore
        self.backbone.fc = nn.Identity()                    # type: ignore

        # 2. 定义新的分类器 (用于小样本微调)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # 3. 加载 DANN 预训练权重
        self.load_dann_backbone(dann_model_path)

        # 4. 冻结 Backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return features, logits
    
    def load_dann_backbone(self, path):
        # 注意：由于 DANN 模型保存时包含 tx_classifier 和 rx_discriminator
        # 我们只加载以 'backbone.' 开头的权重
        state_dict = torch.load(path)
        backbone_dict = {}
        for k, v in state_dict.items():
            if k.startswith('backbone'):
                backbone_dict[k.replace('backbone.', '')] = v
        
        self.backbone.load_state_dict(backbone_dict)
        print(f"Successfully loaded pre-trained DANN backbone from {path}")
