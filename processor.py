"""
Combinations-Thesis - Veri Isleme
Combinations klasöründeki CSV'lerden ozellik cikarir,
her birini (base_type, anomaly_type) ile etiketler.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import warnings
import random
from collections import defaultdict
from tqdm import tqdm

from config import (
    COMBINATIONS_DIR, PROCESSED_DATA_DIR, WINDOW_FRACTION,
    COMBO_FOLDER_MAP, BASE_LABELS, ANOMALY_LABELS,
    MAX_SAMPLES_PER_COMBO, MIN_SERIES_LENGTH, RANDOM_STATE
)

warnings.filterwarnings('ignore')
random.seed(RANDOM_STATE)


# ---------------------------------------------------------------
# Ozellik cikarma
# ---------------------------------------------------------------
def _skewness(data: np.ndarray) -> float:
    n, mean, std = len(data), np.mean(data), np.std(data)
    if std == 0 or n < 3:
        return 0.0
    return (n / ((n-1)*(n-2))) * np.sum(((data - mean) / std) ** 3)


def _kurtosis(data: np.ndarray) -> float:
    n, mean, std = len(data), np.mean(data), np.std(data)
    if std == 0 or n < 4:
        return 0.0
    return (n*(n+1)/((n-1)*(n-2)*(n-3))) * np.sum(((data-mean)/std)**4) \
           - (3*(n-1)**2 / ((n-2)*(n-3)))


def _autocorr(data: np.ndarray, lag: int) -> float:
    if lag >= len(data) or lag < 1:
        return 0.0
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2)
    if c0 == 0:
        return 0.0
    return np.sum((data[:-lag]-mean) * (data[lag:]-mean)) / c0


def _count_peaks(data: np.ndarray) -> int:
    if len(data) < 3:
        return 0
    return int(np.sum((data[1:-1] > data[:-2]) & (data[1:-1] > data[2:])))


def _zero_crossing(data: np.ndarray) -> float:
    if len(data) < 2:
        return 0.0
    return float(np.sum(np.diff(np.sign(data)) != 0) / (len(data) - 1))


def _window_features(win: np.ndarray) -> Optional[Dict]:
    """38 istatistiksel ozellik cikar (tek pencere)."""
    if len(win) < 2:
        return None
    f = {}
    try:
        f['mean']   = np.mean(win)
        f['std']    = np.std(win)
        f['var']    = np.var(win)
        f['min']    = np.min(win)
        f['max']    = np.max(win)
        f['range']  = f['max'] - f['min']
        f['q25']    = np.percentile(win, 25)
        f['median'] = np.median(win)
        f['q75']    = np.percentile(win, 75)
        f['iqr']    = f['q75'] - f['q25']
        f['skewness'] = _skewness(win)
        f['kurtosis'] = _kurtosis(win)
        f['cv']     = f['std'] / (abs(f['mean']) + 1e-10)

        d1 = np.diff(win)
        f['diff1_mean'] = np.mean(d1)
        f['diff1_std']  = np.std(d1)
        f['diff1_var']  = np.var(d1)
        d2 = np.diff(d1) if len(d1) > 1 else np.array([0.0])
        f['diff2_mean'] = np.mean(d2)
        f['diff2_std']  = np.std(d2)

        rw = max(2, len(win) // 5)
        if rw < len(win):
            rm = np.array([np.mean(win[i:i+rw]) for i in range(len(win)-rw+1)])
            rs = np.array([np.std(win[i:i+rw])  for i in range(len(win)-rw+1)])
            f['rolling_mean_std']   = np.std(rm)
            f['rolling_mean_range'] = float(np.max(rm) - np.min(rm))
            f['rolling_std_mean']   = np.mean(rs)
            f['rolling_std_std']    = np.std(rs)
        else:
            f['rolling_mean_std']   = 0.0
            f['rolling_mean_range'] = 0.0
            f['rolling_std_mean']   = f['std']
            f['rolling_std_std']    = 0.0

        half = len(win) // 2
        if half > 0:
            h1, h2 = win[:half], win[half:]
            f['half_mean_diff']  = float(np.mean(h2) - np.mean(h1))
            f['half_std_diff']   = float(np.std(h2)  - np.std(h1))
            f['half_mean_ratio'] = float(np.mean(h2) / (np.mean(h1) + 1e-10))
        else:
            f['half_mean_diff']  = 0.0
            f['half_std_diff']   = 0.0
            f['half_mean_ratio'] = 1.0

        f['autocorr_lag1']      = _autocorr(win, 1)
        f['autocorr_lag10']     = _autocorr(win, min(10, len(win)-1))
        f['num_peaks']          = float(_count_peaks(win))
        f['zero_crossing_rate'] = _zero_crossing(win - np.mean(win))
    except Exception:
        return None
    return f


def extract_features(data: np.ndarray) -> Optional[np.ndarray]:
    """
    L/5 sliding window ile tum seriden ozellik cikar.
    Her pencereden 38 ozellik, pencereler arasi mean/std agregasyonu.
    """
    if len(data) < MIN_SERIES_LENGTH:
        return None

    win_size = max(10, int(len(data) * WINDOW_FRACTION))
    step     = max(1, win_size // 2)

    windows_feat: List[Dict] = []
    for start in range(0, len(data) - win_size + 1, step):
        wf = _window_features(data[start:start + win_size])
        if wf:
            windows_feat.append(wf)

    if not windows_feat:
        return None

    if len(windows_feat) == 1:
        return np.array(list(windows_feat[0].values()), dtype=float)

    # Birden fazla pencere varsa mean + std ile agregasyon
    keys = list(windows_feat[0].keys())
    agg = []
    for k in keys:
        vals = [w[k] for w in windows_feat]
        agg.append(np.mean(vals))
        agg.append(np.std(vals))
    return np.array(agg, dtype=float)


# ---------------------------------------------------------------
# Klasor tarama ve etiket cikartma
# ---------------------------------------------------------------
def _parse_labels(combo_folder_name: str) -> Optional[Tuple[str, str]]:
    key = combo_folder_name.strip().lower()
    return COMBO_FOLDER_MAP.get(key, None)


def scan_combinations() -> List[Tuple[Path, str, str]]:
    """
    Combinations/ dizinini tara, her CSV icin (path, base_type, anomaly_type) dondur.
    """
    all_items: List[Tuple[Path, str, str]] = []
    skipped_dirs = []

    for base_dir in sorted(COMBINATIONS_DIR.iterdir()):
        if not base_dir.is_dir():
            continue
        for combo_dir in sorted(base_dir.iterdir()):
            if not combo_dir.is_dir():
                continue
            labels = _parse_labels(combo_dir.name)
            if labels is None:
                skipped_dirs.append(f"{base_dir.name}/{combo_dir.name}")
                continue
            base_type, anomaly_type = labels
            csvs = list(combo_dir.rglob("*.csv"))
            for csv in csvs:
                all_items.append((csv, base_type, anomaly_type))

    if skipped_dirs:
        print(f"  [WARN] Etiket bulunamayan {len(skipped_dirs)} klasor atlandı:")
        for d in skipped_dirs:
            print(f"    - {d}")
    return all_items


def balance_dataset(items: List[Tuple]) -> List[Tuple]:
    """Her kombinasyon icin MAX_SAMPLES_PER_COMBO ornegini secer."""
    grouped = defaultdict(list)
    for item in items:
        key = f"{item[1]}+{item[2]}"
        grouped[key].append(item)

    balanced = []
    print("\nKombinasyon basi ornek sayisi:")
    for key in sorted(grouped.keys()):
        group = grouped[key]
        if len(group) > MAX_SAMPLES_PER_COMBO:
            group = random.sample(group, MAX_SAMPLES_PER_COMBO)
        print(f"  {key}: {len(group)}")
        balanced.extend(group)
    return balanced


# ---------------------------------------------------------------
# Ana isleme fonksiyonu
# ---------------------------------------------------------------
def process_and_save():
    print("=" * 60)
    print("Combinations verisi isleniyor...")
    print("=" * 60)

    # 1. Dosyalari tara
    all_items = scan_combinations()
    print(f"\nToplam CSV: {len(all_items)}")

    # 2. Dengele
    balanced = balance_dataset(all_items)
    print(f"\nDengeleme sonrasi: {len(balanced)} ornek")

    # 3. Ozellik cikar
    X, y_base, y_anomaly = [], [], []
    failed = 0

    for csv_path, base_type, anomaly_type in tqdm(balanced, desc="Ozellik cikariliyor"):
        try:
            df   = pd.read_csv(csv_path, usecols=['data'])
            data = df['data'].dropna().values.astype(float)
            feat = extract_features(data)
            if feat is None:
                failed += 1
                continue
            X.append(feat)
            y_base.append(BASE_LABELS.index(base_type))
            y_anomaly.append(ANOMALY_LABELS.index(anomaly_type))
        except Exception as e:
            failed += 1

    print(f"\nBasarili: {len(X)}  |  Basarisiz/atlanan: {failed}")

    if not X:
        print("HATA: Hic ornek islenmedi!")
        return

    # Farkli uzunluktaki vektörleri doldur (padding)
    max_len = max(len(x) for x in X)
    X_padded = np.array([
        np.pad(x, (0, max_len - len(x))) if len(x) < max_len else x
        for x in X
    ], dtype=float)

    y_base    = np.array(y_base,    dtype=int)
    y_anomaly = np.array(y_anomaly, dtype=int)

    # 4. Kaydet
    PROCESSED_DATA_DIR.mkdir(exist_ok=True)
    np.save(PROCESSED_DATA_DIR / 'X.npy',        X_padded)
    np.save(PROCESSED_DATA_DIR / 'y_base.npy',   y_base)
    np.save(PROCESSED_DATA_DIR / 'y_anomaly.npy', y_anomaly)

    print(f"\nKaydedildi: {PROCESSED_DATA_DIR}")
    print(f"Ozellik matrisi boyutu: {X_padded.shape}")
    print(f"Base labels dagılımı: {dict(zip(BASE_LABELS, np.bincount(y_base, minlength=len(BASE_LABELS))))}")
    print(f"Anomaly labels dagılımı: {dict(zip(ANOMALY_LABELS, np.bincount(y_anomaly, minlength=len(ANOMALY_LABELS))))}")

    return X_padded, y_base, y_anomaly


if __name__ == "__main__":
    process_and_save()
