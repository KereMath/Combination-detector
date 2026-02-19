"""
Combinations-Thesis - Ana Calistirma Scripti
Kullanim:
    python main.py          # islem + egitim
    python main.py --force  # islemi yeniden yap (cache sil)
"""
import sys
from pathlib import Path
from config import PROCESSED_DATA_DIR

def main():
    force = '--force' in sys.argv
    data_ready = (PROCESSED_DATA_DIR / 'X.npy').exists()

    if force or not data_ready:
        print(">> Adim 1/2: Veri isleniyor...")
        from processor import process_and_save
        process_and_save()
    else:
        import numpy as np
        X = np.load(PROCESSED_DATA_DIR / 'X.npy')
        print(f">> Adim 1/2: Onceden islenmis veri bulundu -> {X.shape}")

    print("\n>> Adim 2/2: Model egitimi baslıyor...")
    from trainer import train_and_evaluate
    train_and_evaluate()

    print("\nTamamlandi.")

if __name__ == "__main__":
    main()
