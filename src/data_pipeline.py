"""Construção do dataset de dígitos a partir do WaterMeters (Partes 1 a 3).

Fluxo: polígono do visor -> faixa retificada 384x64 -> orientação corrigida -> 8 recortes 48x64.
"""
import ast
import re
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

STRIP_W, STRIP_H = 384, 64
N_DIGITS, CROP_W = 8, 48
N_INT, N_DEC = 5, 3
NAME_RE = re.compile(r"id_(\d+)_value_(.*)\.jpg$")


# ---------------------------------------------------------------- aquisição
def check_integrity(root: Path, expected_images: int = 1244) -> dict:
    """Confere a estrutura do download: data.csv + images/ + masks/ + collage/."""
    root = Path(root)
    df = pd.read_csv(root / "data.csv")
    report = {"n_rows_csv": len(df)}
    for sub in ("images", "masks", "collage"):
        names = {p.name for p in (root / sub).glob("*.jpg")}
        report[f"n_{sub}"] = len(names)
        report[f"csv_sem_arquivo_em_{sub}"] = len(set(df.photo_name) - names)
        report[f"arquivo_sem_csv_em_{sub}"] = len(names - set(df.photo_name))
    report["photo_name_duplicado"] = int(df.photo_name.duplicated().sum())
    corrupt = []
    for n in df.photo_name:
        im = cv2.imread(str(root / "images" / n))
        if im is None:
            corrupt.append(n)
    report["imagens_ilegiveis"] = len(corrupt)
    report["ok"] = (
        report["n_rows_csv"] == expected_images
        and report["n_images"] == report["n_masks"] == report["n_collage"] == expected_images
        and not corrupt
        and all(report[k] == 0 for k in report if k.startswith(("csv_sem", "arquivo_sem")))
    )
    return report


def extract_zip_if_needed(zip_path: Path, data_dir: Path) -> None:
    """Alternativa ao download via Kaggle API: extrai o archive.zip já baixado."""
    if (Path(data_dir) / "WaterMeters" / "data.csv").exists():
        return
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(data_dir)


# ------------------------------------------------------------------ rótulos
def parse_name(photo_name: str):
    """id_53_value_595_825.jpg -> (53, '595', '825'). O nome guarda o valor como string exata."""
    m = NAME_RE.match(photo_name)
    image_id, val = int(m.group(1)), m.group(2)
    ip, _, dp = val.partition("_")
    return image_id, ip, dp


def label_8_digits(photo_name: str):
    """Rótulo de 8 dígitos (5 inteiros + 3 decimais), ou None se a leitura não couber nesse formato.

    Inteiros são completados com zeros à esquerda; decimais, com zeros à direita
    (o nome do arquivo omite zeros finais: 567_0 -> 567.000).
    """
    _, ip, dp = parse_name(photo_name)
    if len(ip) > N_INT or len(dp) > N_DEC:
        return None
    return ip.zfill(N_INT) + dp.ljust(N_DEC, "0")


# --------------------------------------------------------------- retificação
def polygon_px(location: str, w: int, h: int) -> np.ndarray:
    pts = ast.literal_eval(location)["data"]
    return np.array([[p["x"] * w, p["y"] * h] for p in pts], np.float32)


def _ordered_quad(poly: np.ndarray) -> np.ndarray:
    """4 cantos em sentido horário, começando de modo que a aresta p0->p1 seja a mais longa.

    Polígonos de 4 pontos são usados como estão (retificam perspectiva). Os demais (6 ou 8 pontos)
    são aproximados pelo retângulo de área mínima.
    """
    quad = poly if len(poly) == 4 else cv2.boxPoints(cv2.minAreaRect(poly))
    c = quad.mean(0)
    ang = np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0])  # eixo y para baixo: crescente = horário
    quad = quad[np.argsort(ang)]
    e = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)  # e[i] = |p_i -> p_i+1|
    if e[0] + e[2] < e[1] + e[3]:
        quad = np.roll(quad, -1, axis=0)
    return quad.astype(np.float32)


def red_score(bgr: np.ndarray) -> float:
    """Intensidade do vermelho na região: percentil 95 de R - max(G, B).

    Mais robusta que um limiar HSV fixo: dígitos vermelhos desbotados ou em fundo escuro ainda
    destacam-se do cinza/branco/preto, que têm R ~ G ~ B.
    """
    b, g, r = (bgr[..., i].astype(np.int16) for i in range(3))
    return float(np.percentile(np.clip(r - np.maximum(g, b), 0, None), 95))


def rectify_strip(img_bgr: np.ndarray, poly_px: np.ndarray, min_red_gap: float = 8.0):
    """Retifica o visor em uma faixa 384x64 com os dígitos na ordem de leitura.

    Retorna (faixa, info). `info` registra o que foi corrigido:
      vertical   - eixo longo do polígono mais vertical que horizontal (faixa girada 90 graus)
      flipped    - faixa girada 180 graus porque o vermelho (decimais) estava à esquerda
      orient_ok  - False se não houve evidência de cor para decidir a orientação (nada foi corrigido)
    """
    quad = _ordered_quad(poly_px)
    dst = np.array([[0, 0], [STRIP_W, 0], [STRIP_W, STRIP_H], [0, STRIP_H]], np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    strip = cv2.warpPerspective(img_bgr, M, (STRIP_W, STRIP_H), flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
    d = quad[1] - quad[0]
    vertical = bool(abs(d[1]) > abs(d[0]))
    band = strip[12:STRIP_H - 12]  # ignora molduras/bordas vermelhas junto às extremidades da faixa
    left = red_score(band[:, : 3 * CROP_W])
    right = red_score(band[:, -3 * CROP_W:])
    orient_ok = abs(left - right) >= min_red_gap
    flipped = bool(orient_ok and left > right)
    if flipped:
        strip = cv2.rotate(strip, cv2.ROTATE_180)
    aspect = float(np.linalg.norm(quad[1] - quad[0]) / max(np.linalg.norm(quad[3] - quad[0]), 1e-6))
    info = dict(vertical=vertical, flipped=flipped, orient_ok=bool(orient_ok), red_left=left,
                red_right=right, aspect=aspect, n_poly_pts=int(len(poly_px)))
    return strip, info


def split_crops(strip: np.ndarray):
    return [strip[:, i * CROP_W:(i + 1) * CROP_W] for i in range(N_DIGITS)]


# ------------------------------------------------------------ dataset final
def build_digit_dataset(raw_root: Path, splits_csv: Path, out_dir: Path, strips_dir: Path | None = None):
    """Gera digits/{split}/{digito}/{image_id}_{pos}.png e digits/index.csv.

    Retorna (index_df, strips_info_df, resumo).
    """
    raw_root, out_dir = Path(raw_root), Path(out_dir)
    data = pd.read_csv(raw_root / "data.csv")
    splits = pd.read_csv(splits_csv)
    split_of = dict(zip(splits.photo_name, splits.split))

    rows, infos, discarded = [], [], []
    for r in data.itertuples():
        lab = label_8_digits(r.photo_name)
        if lab is None or r.photo_name not in split_of:
            discarded.append((r.photo_name, "formato fora de 5+3" if lab is None else "ausente no splits.csv"))
            continue
        img = cv2.imread(str(raw_root / "images" / r.photo_name))
        h, w = img.shape[:2]
        strip, info = rectify_strip(img, polygon_px(r.location, w, h))
        image_id = parse_name(r.photo_name)[0]
        split = split_of[r.photo_name]
        info.update(photo_name=r.photo_name, image_id=image_id, split=split, label=lab)
        infos.append(info)
        if strips_dir is not None:
            Path(strips_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(strips_dir) / f"{image_id}.png"), strip)
        for pos, crop in enumerate(split_crops(strip)):
            digit = lab[pos]
            p = out_dir / split / digit
            p.mkdir(parents=True, exist_ok=True)
            rel = f"{split}/{digit}/{image_id}_{pos}.png"
            cv2.imwrite(str(out_dir / rel), crop)
            rows.append((image_id, r.photo_name, pos, int(digit), split, f"digits/{rel}"))

    index = pd.DataFrame(rows, columns=["image_id", "photo_name", "position", "digit", "split", "crop_path"])
    index.to_csv(out_dir / "index.csv", index=False)
    infos = pd.DataFrame(infos)
    summary = dict(
        n_valid=len(infos), n_discarded=len(discarded), discarded=discarded, n_crops=len(index),
        n_vertical=int(infos.vertical.sum()), n_flipped=int(infos.flipped.sum()),
        n_corrected=int((infos.vertical | infos.flipped).sum()), n_orient_undecided=int((~infos.orient_ok).sum()),
        leakage=int((index.groupby("photo_name").split.nunique() > 1).sum()),
    )
    return index, infos, summary
