"""Загрузка модели, признаков и метаданных."""
import joblib
from pathlib import Path


# app/src/data_loader.py → parent = src → parent.parent = app → models/
MODEL_DIR = Path(__file__).resolve().parent.parent / 'models'


def _require(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    return path


def load_model():
    """Обученная модель (Random Forest)."""
    return joblib.load(_require(MODEL_DIR / 'best_model.pkl'))


def load_feature_columns():
    """Список признаков, на которых обучалась модель."""
    return joblib.load(_require(MODEL_DIR / 'feature_columns.pkl'))


def get_model_info():
    return {
        'name': 'Random Forest',
        'algorithm': 'бэггинг деревьев решений',
        'r2':   0.9669,
        'mae':  104_329,
        'rmse': 337_667,
        'mape': 30.88,
        'train_size': 199_767,
        'test_size':  49_942,
        'n_features': 53,
        'period': '2018–2019',
    }