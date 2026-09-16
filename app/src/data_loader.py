import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / 'models'


def load_model():
    """Загружает обученную модель XGBoost."""
    model_path = MODEL_DIR / 'best_model.pkl'
    return joblib.load(model_path)


def load_feature_columns():
    """Загружает список признаков, на которых обучалась модель."""
    cols_path = MODEL_DIR / 'feature_columns.pkl'
    return joblib.load(cols_path)


def get_model_info():
    """Возвращает метаданные для отображения в UI."""
    return {
        'name': 'XGBoost',
        'r2': 0.947,
        'mape': 3.81,
        'train_size': 199_771,
        'period': '2018-2019'
    }