import pandas as pd


# Допустимые значения для каждого поля
ALLOWED_VALUES = {
    'sub_type': ['Daire', 'Villa', 'Müstakil Ev', 'Rezidans', 'Yazlık', 'Другие'],
    'city': ['İstanbul', 'Ankara', 'İzmir', 'Antalya', 'Aydın', 'Muğla', 'Mersin', 'Другие'],
    'heating_type': ['Kombi (Doğalgaz)', 'Merkezi Sistem', 'Klima', 'Yerden Isıtma', 'Soba (Kömür)', 'Другие'],
    'building_age': ['0', '1', '2', '3', '4', '5', '6-10 arası', '11-15 arası', '16+', 'Другие'],
    'room_count': ['1+0', '1+1', '2+1', '3+1', '3+2', '4+2', '5+2'],
}


def validate_input(user_input: dict) -> dict:
    """Проверяет, что все поля в допустимых диапазонах."""
    errors = []
    
    if not (10 <= user_input.get('total_area', 0) <= 500):
        errors.append('Площадь должна быть от 10 до 500 м²')
    
    for field, allowed in ALLOWED_VALUES.items():
        if user_input.get(field) not in allowed:
            errors.append(f'Недопустимое значение для {field}')
    
    if errors:
        raise ValueError('; '.join(errors))
    
    return user_input


def encode_input(user_input: dict, feature_columns: list) -> pd.DataFrame:
    df = pd.DataFrame([user_input])
    
    df = pd.get_dummies(
        df,
        columns=['sub_type', 'city', 'heating_type', 'room_count', 'building_age'],
        drop_first=True,
        dtype=int
    )
    
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    return df[feature_columns]