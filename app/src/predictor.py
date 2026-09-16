import numpy as np
from .data_loader import load_model, load_feature_columns, get_model_info
from .preprocessing import validate_input, encode_input

class PricePredictor:
    """Фасад для работы с моделью прогнозирования."""
    
    def __init__(self):
        self.model = load_model()
        self.feature_columns = load_feature_columns()
        self.info = get_model_info()
    
    def predict(self, user_input: dict) -> dict:
        """Принимает словарь с характеристиками объекта, возвращает словарь с результатом прогноза."""
        validate_input(user_input)
        
        X = encode_input(user_input, self.feature_columns)
        log_price = self.model.predict(X)[0]
        price = float(np.expm1(log_price))
        
        mape = self.info['mape'] / 100
        return {
            'price': price,
            'lower_bound': price * (1 - mape),
            'upper_bound': price * (1 + mape),
            'mape_pct': self.info['mape'],
            'r2': self.info['r2'],
        }


_predictor = None


def get_predictor() -> PricePredictor:
    global _predictor
    if _predictor is None:
        _predictor = PricePredictor()
    return _predictor