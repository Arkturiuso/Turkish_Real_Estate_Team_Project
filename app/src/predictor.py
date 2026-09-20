"""Фасад для работы с моделью прогнозирования."""
import numpy as np

from .data_loader import load_model, load_feature_columns, get_model_info
from .preprocessing import validate_input, encode_input


class PricePredictor:
    def __init__(self):
        self.model = load_model()
        self.feature_columns = load_feature_columns()
        self.info = get_model_info()

    def predict(self, user_input: dict) -> dict:
        validate_input(user_input)

        X = encode_input(user_input, self.feature_columns)
        log_price = float(self.model.predict(X)[0])
        price = float(np.expm1(log_price))

        # Ориентировочный интервал ±MAPE (пока не сохранены квантили остатков)
        m = self.info['mape'] / 100.0
        return {
            'price':         price,
            'lower_bound':   max(0.0, price * (1 - m)),
            'upper_bound':   price * (1 + m),
            'r2':            self.info['r2'],
            'mae':           self.info['mae'],
            'model_name':    self.info['name'],
        }


_predictor = None


def get_predictor() -> PricePredictor:
    global _predictor
    if _predictor is None:
        _predictor = PricePredictor()
    return _predictor