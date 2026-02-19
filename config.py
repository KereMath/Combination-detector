"""
Combinations-Thesis - Configuration
Amac: Combinations klasoründeki zaman serilerinden
      hem base tipi hem anomali tipini dogru tahmin etmek.
"""
from pathlib import Path

BASE_DIR        = Path(__file__).parent
COMBINATIONS_DIR = Path(r"C:\Users\user\Desktop\STATIONARY\Combinations")
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
MODELS_DIR      = BASE_DIR / "trained_models"
RESULTS_DIR     = BASE_DIR / "results"

# -------------------------------------------------------------------
# Window stratejisi (L/5 - tum diger detector'larda en iyi)
# -------------------------------------------------------------------
WINDOW_FRACTION = 0.2  # L/5

# -------------------------------------------------------------------
# Label tanimlari
# -------------------------------------------------------------------
BASE_LABELS   = ["cubic", "damped", "exponential", "linear", "quadratic"]
ANOMALY_LABELS = ["collective_anomaly", "mean_shift", "point_anomaly",
                  "trend_shift", "variance_shift"]

# -------------------------------------------------------------------
# Klasor adi → (base_type, anomaly_type) eslestirmesi
# Anahtar: kucuk harfe indirilmis klasor adi
# -------------------------------------------------------------------
COMBO_FOLDER_MAP = {
    # Cubic
    "cubic + mean shift":              ("cubic", "mean_shift"),
    "cubic + point anomaly":           ("cubic", "point_anomaly"),
    "cubic + variance shift":          ("cubic", "variance_shift"),
    "cubic_collective_anomaly":        ("cubic", "collective_anomaly"),
    # Damped
    "damped + collective anomaly":     ("damped", "collective_anomaly"),
    "damped + mean shift":             ("damped", "mean_shift"),
    "damped + point anomaly":          ("damped", "point_anomaly"),
    "damped + variance shift":         ("damped", "variance_shift"),
    # Exponential
    "exponential + mean shift":        ("exponential", "mean_shift"),
    "exponential_collective_anomaly":  ("exponential", "collective_anomaly"),
    "exponential_point_anomaly":       ("exponential", "point_anomaly"),
    "exponential_variance_shift":      ("exponential", "variance_shift"),
    # Linear
    "linear + collective anomaly":     ("linear", "collective_anomaly"),
    "linear + mean shift":             ("linear", "mean_shift"),
    "linear + point anomaly":          ("linear", "point_anomaly"),
    "linear + trend shift":            ("linear", "trend_shift"),
    "linear + variance shift":         ("linear", "variance_shift"),
    # Quadratic
    "quadratic + collective anomaly":  ("quadratic", "collective_anomaly"),
    "quadratic + collective anomaly ": ("quadratic", "collective_anomaly"),
    "quadratic + mean shift":          ("quadratic", "mean_shift"),
    "quadratic + point anomaly":       ("quadratic", "point_anomaly"),
    "quadratic + variance shift":      ("quadratic", "variance_shift"),
}

# -------------------------------------------------------------------
# Egitim parametreleri
# -------------------------------------------------------------------
MAX_SAMPLES_PER_COMBO = 800   # Dengeleme icin kombinasyon basina max ornek
TEST_SIZE             = 0.20
VALIDATION_SIZE       = 0.10
RANDOM_STATE          = 42
MIN_SERIES_LENGTH     = 50
CHUNK_SIZE            = 10000
