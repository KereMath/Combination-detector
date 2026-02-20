"""
Combinations-Thesis - Veri Isleme
Combinations klasöründeki CSV'lerden tsfresh ile ozellik cikarir,
her birini (base_type, anomaly_type) ile etiketler.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List
import warnings
import random
from collections import defaultdict
from tqdm import tqdm

from tsfresh import extract_features as tsfresh_extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters

from config import (
    COMBINATIONS_DIR, PROCESSED_DATA_DIR, COMBO_FOLDER_MAP,
    BASE_LABELS, ANOMALY_LABELS,
    MAX_SAMPLES_PER_COMBO, MIN_SERIES_LENGTH, RANDOM_STATE
)

warnings.filterwarnings('ignore')
random.seed(RANDOM_STATE)


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
    print("Combinations verisi isleniyor (tsfresh)...")
    print("=" * 60)

    # 1. Dosyalari tara
    all_items = scan_combinations()
    print(f"\nToplam CSV: {len(all_items)}")

    # 2. Dengele
    balanced = balance_dataset(all_items)
    print(f"\nDengeleme sonrasi: {len(balanced)} ornek")

    # 3. CSV'leri yükle, long-format DataFrame olustur
    series_dfs: List[pd.DataFrame] = []
    labels: List[Tuple[str, str]] = []
    sid = 0
    failed = 0

    for csv_path, base_type, anomaly_type in tqdm(balanced, desc="CSV okunuyor"):
        try:
            df   = pd.read_csv(csv_path, usecols=['data'])
            data = df['data'].dropna().values.astype(float)
            if len(data) < MIN_SERIES_LENGTH:
                failed += 1
                continue
            series_dfs.append(pd.DataFrame({
                'id':    sid,
                'time':  np.arange(len(data)),
                'value': data,
            }))
            labels.append((base_type, anomaly_type))
            sid += 1
        except Exception:
            failed += 1

    print(f"\nBasarili: {sid}  |  Basarisiz/atlanan: {failed}")

    if not series_dfs:
        print("HATA: Hic ornek islenmedi!")
        return

    combined_df = pd.concat(series_dfs, ignore_index=True)

    # 4. tsfresh ile toplu ozellik cikarimi
    print(f"\ntsfresh ozellik cikarimi basliyor ({sid} seri, EfficientFCParameters)...")
    print("Not: Bu adim veri buyuklugune gore 5-30 dk surebilir.\n")

    X_feat = tsfresh_extract_features(
        combined_df,
        column_id='id',
        column_sort='time',
        column_value='value',
        default_fc_parameters=EfficientFCParameters(),
    )

    # NaN / Inf temizle
    impute(X_feat)

    X         = X_feat.values
    y_base    = np.array([BASE_LABELS.index(l[0])    for l in labels], dtype=int)
    y_anomaly = np.array([ANOMALY_LABELS.index(l[1]) for l in labels], dtype=int)

    # 5. Kaydet
    PROCESSED_DATA_DIR.mkdir(exist_ok=True)
    np.save(PROCESSED_DATA_DIR / 'X.npy',         X)
    np.save(PROCESSED_DATA_DIR / 'y_base.npy',    y_base)
    np.save(PROCESSED_DATA_DIR / 'y_anomaly.npy', y_anomaly)

    # Ozellik isimlerini de kaydet (trainer'da debug icin faydali)
    feature_names = list(X_feat.columns)
    import json
    with open(PROCESSED_DATA_DIR / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f)

    print(f"\nKaydedildi: {PROCESSED_DATA_DIR}")
    print(f"Ozellik matrisi boyutu: {X.shape}")
    print(f"Base labels dagılımı:   {dict(zip(BASE_LABELS, np.bincount(y_base,    minlength=len(BASE_LABELS))))}")
    print(f"Anomaly labels dagılımı:{dict(zip(ANOMALY_LABELS, np.bincount(y_anomaly, minlength=len(ANOMALY_LABELS))))}")

    return X, y_base, y_anomaly


if __name__ == "__main__":
    process_and_save()
