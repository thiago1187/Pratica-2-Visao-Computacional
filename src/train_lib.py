"""Modelos, treino e avaliação (Partes 4 e 5). Protocolo idêntico para os quatro modelos."""
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2

SEED = 42
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # estatísticas do ImageNet

# Protocolo oficial (README do professor). batch_size = valor para MPS/CPU.
PROTOCOL = {
    "smallcnn": dict(input=(64, 48), batch=128, lr_head=1e-3, lr_backbone=None, epochs=10, pretrained=False),
    "alexnet": dict(input=(128, 128), batch=64, lr_head=1e-3, lr_backbone=1e-4, epochs=8, pretrained=True),
    "vgg16_bn": dict(input=(128, 128), batch=16, lr_head=1e-3, lr_backbone=1e-5, epochs=8, pretrained=True),
    "resnet18": dict(input=(128, 128), batch=64, lr_head=1e-3, lr_backbone=1e-4, epochs=8, pretrained=True),
}
LABEL_SMOOTHING, PATIENCE, N_CLASSES = 0.1, 3, 10


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


# ------------------------------------------------------------------- dados
def load_crops(index: pd.DataFrame, root: Path) -> torch.Tensor:
    """Carrega os recortes 48x64 em um tensor uint8 (N, 3, 64, 48), RGB."""
    arr = np.stack([cv2.cvtColor(cv2.imread(str(Path(root) / p)), cv2.COLOR_BGR2RGB) for p in index.crop_path])
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


class CropDataset(Dataset):
    """Augmentation (só no treino) sobre o recorte 48x64; normalização ImageNet. Sem flip."""

    def __init__(self, x_uint8, y, train: bool):
        self.x, self.y, self.train = x_uint8, torch.as_tensor(y, dtype=torch.long), train
        self.aug = v2.Compose([
            v2.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            v2.ColorJitter(brightness=0.3, contrast=0.3),
        ])
        self.norm = v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.x[i]
        if self.train:
            x = self.aug(x)
        return self.norm(x), self.y[i]


def make_loader(x, y, train: bool, batch: int, workers: int = 4):
    g = torch.Generator().manual_seed(SEED)
    return DataLoader(CropDataset(x, y, train), batch_size=batch, shuffle=train, generator=g,
                      num_workers=workers, persistent_workers=workers > 0, drop_last=False)


# ----------------------------------------------------------------- modelos
class SmallCNN(nn.Module):
    """CNN pequena treinada do zero: 3 blocos (conv-BN-ReLU x2 + maxpool) + GAP + linear."""

    def __init__(self, n_classes=N_CLASSES):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2))

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class AdaptiveAvgPool2dAnyDevice(nn.Module):
    """Equivalente exato de nn.AdaptiveAvgPool2d, via matrizes de pooling.

    O MPS não implementa o pooling adaptativo quando a saída é maior que a entrada (AlexNet: 3x3 -> 6x6;
    VGG16: 4x4 -> 7x7 com entradas 128x128). A média de cada janela adaptativa é uma multiplicação por matriz.
    """

    def __init__(self, output_size):
        super().__init__()
        self.out = output_size if isinstance(output_size, tuple) else (output_size, output_size)
        self._cache = {}

    def _mat(self, n_in, n_out, device, dtype):
        key = (n_in, n_out, device, dtype)
        if key not in self._cache:
            m = torch.zeros(n_out, n_in)
            for i in range(n_out):
                a, b = (i * n_in) // n_out, -((-(i + 1) * n_in) // n_out)  # floor(i*in/out), ceil((i+1)*in/out)
                m[i, a:b] = 1.0 / (b - a)
            self._cache[key] = m.to(device=device, dtype=dtype)
        return self._cache[key]

    def forward(self, x):
        ph = self._mat(x.shape[-2], self.out[0], x.device, x.dtype)
        pw = self._mat(x.shape[-1], self.out[1], x.device, x.dtype)
        return torch.einsum("oh,nchw,pw->ncop", ph, x, pw)


def build_model(name: str):
    """Retorna (modelo, parâmetros_da_cabeça, parâmetros_do_backbone).

    Cabeça = última camada linear, substituída por uma nova de 10 classes. Backbone = o restante
    (inclusive as FC intermediárias de AlexNet/VGG16, que são pré-treinadas).
    """
    pre = PROTOCOL[name]["pretrained"]
    if name == "smallcnn":
        m = SmallCNN()
        head = list(m.head[1].parameters())
    elif name == "alexnet":
        m = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1 if pre else None)
        m.classifier[6] = nn.Linear(4096, N_CLASSES)
        m.avgpool = AdaptiveAvgPool2dAnyDevice((6, 6))
        head = list(m.classifier[6].parameters())
    elif name == "vgg16_bn":
        m = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1 if pre else None)
        m.classifier[6] = nn.Linear(4096, N_CLASSES)
        m.avgpool = AdaptiveAvgPool2dAnyDevice((7, 7))
        head = list(m.classifier[6].parameters())
    elif name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pre else None)
        m.fc = nn.Linear(512, N_CLASSES)
        head = list(m.fc.parameters())
    else:
        raise ValueError(name)
    ids = {id(p) for p in head}
    backbone = [p for p in m.parameters() if id(p) not in ids]
    return m, head, backbone


class Wrapped(nn.Module):
    """Redimensiona o recorte para a entrada do modelo (no device) antes do forward."""

    def __init__(self, net, size):
        super().__init__()
        self.net, self.size = net, size

    def forward(self, x):
        if tuple(x.shape[-2:]) != tuple(self.size):
            x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
        return self.net(x)


def n_params_M(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


# ------------------------------------------------------------------ treino
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits, ys = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).float().cpu())
        ys.append(y)
    return torch.cat(logits), torch.cat(ys)


def train_model(name, x_tr, y_tr, x_va, y_va, device, ckpt_dir: Path, workers: int = 4, log=print):
    """Treina `name` com o protocolo oficial. Retorna (history_df, caminho_do_melhor_checkpoint)."""
    cfg = PROTOCOL[name]
    seed_everything()
    net, head, backbone = build_model(name)
    model = Wrapped(net, cfg["input"]).to(device)
    tr_loader = make_loader(x_tr, y_tr, True, cfg["batch"], workers)
    va_loader = make_loader(x_va, y_va, False, cfg["batch"], workers)

    # smallcnn (do zero) não tem LR separado: todo o modelo treina com o LR "da cabeça" (1e-3)
    lr_bb = cfg["lr_backbone"] if cfg["lr_backbone"] is not None else cfg["lr_head"]
    groups = [{"params": head, "lr": cfg["lr_head"]}, {"params": backbone, "lr": lr_bb}]
    max_lrs = [cfg["lr_head"], lr_bb]
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lrs, epochs=cfg["epochs"],
                                                steps_per_epoch=len(tr_loader))
    crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / f"{name}.pt"
    rows, best_f1, best_ep, bad = [], -1.0, 0, 0
    for ep in range(1, cfg["epochs"] + 1):
        freeze = cfg["pretrained"] and ep == 1  # 1ª época: só a cabeça treina
        for p in backbone:
            p.requires_grad_(not freeze)
        model.train()
        if freeze:  # mantém BN/Dropout do backbone em modo de avaliação enquanto congelado
            _set_backbone_eval(model.net, name)
        sync(device)
        t0 = time.perf_counter()
        loss_sum, correct, n = 0.0, 0, 0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
        sync(device)
        seconds = time.perf_counter() - t0  # só treino, sem validação

        lg, yy = predict(model, va_loader, device)
        va_loss = crit(lg, yy).item()
        pred = lg.argmax(1).numpy()
        va_acc = float((pred == yy.numpy()).mean())
        va_f1 = float(f1_score(yy.numpy(), pred, average="macro"))
        rows.append(dict(epoch=ep, train_loss=loss_sum / n, train_acc=correct / n, val_loss=va_loss,
                         val_acc=va_acc, val_f1=va_f1, seconds=seconds))
        log(f"[{name}] ep {ep}/{cfg['epochs']} loss {loss_sum / n:.4f} acc {correct / n:.4f} | "
            f"val_loss {va_loss:.4f} val_acc {va_acc:.4f} val_f1 {va_f1:.4f} | {seconds:.1f}s")
        if va_f1 > best_f1:
            best_f1, best_ep, bad = va_f1, ep, 0
            torch.save(model.net.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= PATIENCE:
                log(f"[{name}] early stopping na época {ep} (melhor: {best_ep})")
                break
    return pd.DataFrame(rows), ckpt


def _set_backbone_eval(net, name):
    """Coloca em eval() tudo exceto a cabeça (BN não atualiza estatísticas; dropout desligado)."""
    net.eval()
    head = {"smallcnn": lambda: net.head[1], "alexnet": lambda: net.classifier[6],
            "vgg16_bn": lambda: net.classifier[6], "resnet18": lambda: net.fc}[name]()
    head.train()


# --------------------------------------------------------------- avaliação
def load_best(name, ckpt, device):
    net, _, _ = build_model(name)
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return Wrapped(net, PROTOCOL[name]["input"]).to(device)


@torch.no_grad()
def latency_ms(model, device, size, n_runs=200, warmup=20):
    """Inferência com batch 1: mediana de `n_runs` execuções após `warmup` de aquecimento."""
    model.eval()
    x = torch.randn(1, 3, *size, device=device)
    for _ in range(warmup):
        model(x)
    sync(device)
    ts = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model(x)
        sync(device)
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def gmacs(model, size) -> float:
    """GMACs por imagem (batch 1) via contador de FLOPs do PyTorch (FLOPs / 2)."""
    from torch.utils.flop_counter import FlopCounterMode
    m = model.to("cpu").eval()
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        m(torch.randn(1, 3, *size))
    return fc.get_total_flops() / 2 / 1e9
