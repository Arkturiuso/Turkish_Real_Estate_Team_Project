import sys
sys.path.append('..')
from app.src.predictor import get_predictor


def test_predict_returns_dict():
    predictor = get_predictor()
    result = predictor.predict({
        'total_area': 100, 'room_count': '2+1', 'floor_num': 3,
        'floors_total_num': 10, 'sub_type': 'Daire', 'city': 'İstanbul',
        'heating_type': 'Kombi (Doğalgaz)', 'building_age': '5'
    })
    assert 'price' in result
    assert result['price'] > 0
    assert result['lower_bound'] < result['price'] < result['upper_bound']
    print("test_predict_returns_dict passed")


def test_validation_rejects_bad_area():
    predictor = get_predictor()
    try:
        predictor.predict({'total_area': 5, 'room_count': '2+1', 'floor_num': 3,
                          'floors_total_num': 10, 'sub_type': 'Daire', 'city': 'İstanbul',
                          'heating_type': 'Kombi (Doğalgaz)', 'building_age': '5'})
        assert False, "Должно было упасть"
    except ValueError:
        print("test_validation_rejects_bad_area passed")


if __name__ == '__main__':
    test_predict_returns_dict()
    test_validation_rejects_bad_area()
    print("\nВсе тесты прошли")