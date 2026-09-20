"""Валидация и кодирование входных данных."""
import numpy as np
import pandas as pd


# Полные списки категорий — включая удалённые референсные классы
# (они дают all-zeros и корректно интерпретируются моделью).
ALLOWED_VALUES = {
    'listing_type': ['Аренда', 'Аренда на день', 'Продажа'],

    'sub_type': [
        'Вилла', 'Дача', 'Другие', 'Квартира', 'Квартира у воды',
        'Особняк / Усадьба / Дом у воды', 'Отдельный дом', 'Резиденция',
        'Сборный дом', 'Фермерский дом', 'Целое здание',
    ],

    'city': [
        'Адана', 'Айдын', 'Анкара', 'Анталья', 'Балыкесир', 'Другие',
        'Измир', 'Мерсин', 'Мугла', 'Стамбул',
    ],

    'heating_type': [
        'Фанкойл', 'Газовый котёл', 'Другие', 'Кондиционер', 'Нет',
        'Печь (уголь)', 'Тёплый пол', 'Центральное', 'Центральное (газ)',
        'Центральное (счётчик тепла)',
    ],

    'building_age': [
        '0', '1', '2', '4', '6-10 лет', '11-15 лет', '16-20 лет',
        '21-25 лет', 'Другие', 'Не указано',
    ],

    'room_count': [
        '1+0', '1+1', '2+1', '3+1', '3+2', '4+1', '4+2',
        '5+1', '5+2', 'Другие',
    ],
}

DUMMY_COLS = ['listing_type', 'sub_type', 'city', 'heating_type', 'building_age', 'room_count']

# Среднее tom из обучающей выборки — модель его видела, пользователь не вводит
TOM_DEFAULT = 57.0


def _parse_rooms(val) -> float:
    """'4+2' → 6.0, '1+0' → 1.0, '2+1' → 3.0."""
    if pd.isna(val):
        return 0.0
    val = str(val).strip()
    if '+' in val:
        parts = val.split('+')
        try:
            return float(parts[0]) + float(parts[1])
        except ValueError:
            return 0.0
    try:
        return float(val)
    except ValueError:
        return 0.0


def validate_input(user_input: dict) -> dict:
    errors = []

    area = user_input.get('total_area', 0)
    if not (10 <= area <= 500):
        errors.append('Площадь должна быть от 10 до 500 м²')

    if user_input.get('floor_num', 0) > user_input.get('floors_total_num', 1):
        errors.append('Этаж не может быть выше общего количества этажей')

    for field, allowed in ALLOWED_VALUES.items():
        if user_input.get(field) not in allowed:
            errors.append(f'Недопустимое значение для "{field}": {user_input.get(field)}')

    if errors:
        raise ValueError('; '.join(errors))

    return user_input


def encode_input(user_input: dict, feature_columns: list) -> pd.DataFrame:
    """Преобразует ввод в DataFrame, согласованный с моделью.

    ВАЖНО:
    - drop_first=False — иначе для одно-строчного DataFrame столбец категории
      будет удалён и все one-hot признаки окажутся нулями.
    - Модель (Random Forest) обучалась на СЫРЫХ данных — scaler НЕ применяется.
    """
    df = pd.DataFrame([user_input])

    df['rooms_num'] = _parse_rooms(user_input.get('room_count', ''))
    df['tom'] = TOM_DEFAULT

    df = pd.get_dummies(df, columns=DUMMY_COLS, drop_first=False, dtype=int)

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    return df[feature_columns]