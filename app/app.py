from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.predictor import get_predictor


st.set_page_config(
    page_title="Прогноз цен на недвижимость",
    layout="wide",
)
st.title("Прогнозирование стоимости недвижимости в Турции")

predictor = get_predictor()

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

RU_PATHS = [
    PROJECT_ROOT / 'real_estate_data_ru.csv',
    Path.cwd() / 'real_estate_data_ru.csv',
    APP_DIR / 'real_estate_data_ru.csv',
]


def _find_file(paths):
    for p in paths:
        if p.exists():
            return p
    return None


@st.cache_data(show_spinner="Загружаю данные для дашборда…")
def load_dashboard_data(path: str):
    """Загружает датасет и группирует редкие категории так же, как при обучении модели.

    Возвращает (df, n_raw_cities, n_raw_subtypes) — сам датасет и число уникальных
    значений в сырых данных (для пояснительных подписей в UI).
    """
    df = pd.read_csv(path)
    if 'price' in df.columns:
        df = df[df['price'] > 0].copy()

    n_raw_cities = df['city'].nunique() if 'city' in df.columns else 0
    n_raw_subtypes = df['sub_type'].nunique() if 'sub_type' in df.columns else 0

    if 'city' in df.columns:
        top_cities = df['city'].value_counts().head(10).index
        df['city'] = df['city'].apply(lambda x: x if x in top_cities else 'Другие')

    if 'sub_type' in df.columns:
        top_sub = df['sub_type'].value_counts().head(10).index
        df['sub_type'] = df['sub_type'].apply(lambda x: x if x in top_sub else 'Другие')

    return df, n_raw_cities, n_raw_subtypes

st.sidebar.header("Параметры объекта")

user_input = {
    'listing_type': st.sidebar.selectbox(
        "Тип сделки", ['Продажа', 'Аренда'], index=0,
        help="Продажа или долгосрочная аренда. Посуточная аренда не поддерживается моделью."),
    'total_area': st.sidebar.number_input(
        "Площадь (м²)", 10.0, 500.0, 100.0, 5.0,
        help="Общая площадь объекта. Допустимый диапазон: 10–500 м²."),
    'room_count': st.sidebar.selectbox(
        "Комнат",
        ['1+0', '1+1', '2+1', '3+1', '3+2', '4+1', '4+2', '5+1', '5+2', 'Другие'],
        index=2,
        help="Планировка: жилые+общие комнаты. «2+1» — 2 спальни и гостиная."),
    'floor_num': st.sidebar.number_input(
        "Этаж", 0, 50, 3,
        help="Этаж, на котором находится объект. 0 — цоколь или первый."),
    'floors_total_num': st.sidebar.number_input(
        "Этажей в доме", 1, 100, 10,
        help="Общая этажность здания."),
    'sub_type': st.sidebar.selectbox(
        "Тип недвижимости",
        ['Вилла', 'Дача', 'Другие', 'Квартира', 'Квартира у воды',
         'Особняк / Усадьба / Дом у воды', 'Отдельный дом', 'Резиденция',
         'Сборный дом', 'Фермерский дом', 'Целое здание'],
        index=3),
    'city': st.sidebar.selectbox(
        "Город",
        ['Адана', 'Айдын', 'Анкара', 'Анталья', 'Балыкесир', 'Другие',
         'Измир', 'Мерсин', 'Мугла', 'Стамбул'],
        index=9,
        help="Город. Модель не различает районы внутри города."),
    'heating_type': st.sidebar.selectbox(
        "Отопление",
        ['Фанкойл', 'Газовый котёл', 'Другие', 'Кондиционер', 'Нет',
         'Печь (уголь)', 'Тёплый пол', 'Центральное', 'Центральное (газ)',
         'Центральное (счётчик тепла)'],
        index=1),
    'building_age': st.sidebar.selectbox(
        "Возраст здания",
        ['0', '1', '2', '4', '6-10 лет', '11-15 лет', '16-20 лет',
         '21-25 лет', 'Другие', 'Не указано'],
        index=0,
        help="Возраст здания от года постройки. «0» — новостройка."),
}


tab_form, tab_dashboard, tab_help = st.tabs(
    ["Прогноз", "Дашборд", "Справка"]
)


# Вкладка 1: Прогноз
with tab_form:
    st.header("Расчёт стоимости объекта")
    st.caption("Заполните форму слева и нажмите «Рассчитать стоимость».")

    if st.button("Рассчитать стоимость", type="primary", use_container_width=True):
        try:
            result = predictor.predict(user_input)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Предсказанная цена", f"{result['price']:,.0f} ₺")
            with col2:
                st.metric("В долларах (≈30 ₺/$)", f"${result['price'] / 30:,.0f}")
            with col3:
                st.metric("В евро (≈33 ₺/€)", f"€{result['price'] / 33:,.0f}")

            st.info(
                f"**Ориентировочный интервал (±{predictor.info['mape']:.1f}% MAPE):**\n\n"
                f"от **{result['lower_bound']:,.0f} ₺** до **{result['upper_bound']:,.0f} ₺**"
            )
            st.caption(
                f"Модель: **{result['model_name']}**  ·  "
                f"R² = **{result['r2']:.4f}**  ·  "
                f"MAE = **{result['mae']:,.0f} ₺**"
            )

            with st.expander("Что именно вы ввели"):
                st.json(user_input)

        except ValueError as e:
            st.error(f"Ошибка валидации: {e}")
        except Exception as e:
            st.error(f"Произошла ошибка при расчёте: {e}")


# Вкладка 2: Дашборд
with tab_dashboard:
    st.header("Дашборд исследования данных")
    st.caption("Интерактивные графики по датасету объявлений о недвижимости (2018–2019).")

    ru_file = _find_file(RU_PATHS)
    if ru_file is None:
        st.warning(
            "Файл `real_estate_data_ru.csv` не найден в корне проекта. "
            "Дашборд недоступен без него."
        )
        df_raw = None
        n_raw_cities = 0
        n_raw_subtypes = 0
    else:
        df_raw, n_raw_cities, n_raw_subtypes = load_dashboard_data(str(ru_file))

    if df_raw is not None:
        st.subheader("Общая статистика")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Всего объектов", f"{len(df_raw):,}")
        with col2:
            st.metric("Средняя цена", f"{df_raw['price'].mean():,.0f} ₺")
        with col3:
            st.metric("Медианная цена", f"{df_raw['price'].median():,.0f} ₺")
        with col4:
            if 'total_area' in df_raw.columns:
                st.metric("Средняя площадь", f"{df_raw['total_area'].mean():.1f} м²")
        with col5:
            if 'city' in df_raw.columns:
                st.metric(
                    "Уникальных городов",
                    f"{df_raw['city'].nunique()}",
                    help=f"В сырых данных было {n_raw_cities} уникальных значений. "
                         f"При обучении модели редкие города объединены в категорию «Другие».",
                )

        st.markdown("---")

        st.subheader("Распределения признаков")
        dist_col1, dist_col2 = st.columns(2)

        with dist_col1:
            st.markdown("##### Распределение цен")
            price_data = df_raw[df_raw['price'] <= df_raw['price'].quantile(0.99)]
            fig_price = px.histogram(
                price_data, x='price', nbins=50,
                title='Распределение цен (без топ-1% выбросов)',
                labels={'price': 'Цена (₺)', 'count': 'Количество'},
                color_discrete_sequence=['#3498db'],
            )
            fig_price.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_price, use_container_width=True)

        with dist_col2:
            if 'total_area' in df_raw.columns:
                st.markdown("##### Распределение площади")
                area_data = df_raw[df_raw['total_area'] <= df_raw['total_area'].quantile(0.99)]
                fig_area = px.histogram(
                    area_data, x='total_area', nbins=50,
                    title='Распределение площади (без топ-1% выбросов)',
                    labels={'total_area': 'Площадь (м²)', 'count': 'Количество'},
                    color_discrete_sequence=['#2ecc71'],
                )
                fig_area.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_area, use_container_width=True)

        st.markdown("---")

        st.subheader("Средние цены по категориям")
        cat_col1, cat_col2 = st.columns(2)

        with cat_col1:
            st.markdown("##### По типу недвижимости")
            if 'sub_type' in df_raw.columns:
                type_avg = (
                    df_raw.groupby('sub_type')['price']
                    .mean().sort_values(ascending=False).head(10)
                    .reset_index()
                )
                fig_type = px.bar(
                    type_avg, x='price', y='sub_type', orientation='h',
                    title='Средняя цена по типам недвижимости',
                    labels={'price': 'Средняя цена (₺)', 'sub_type': ''},
                    color='price', color_continuous_scale='Blues',
                )
                fig_type.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20),
                                       showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig_type, use_container_width=True)

        with cat_col2:
            st.markdown("##### По городам (топ-10)")
            if 'city' in df_raw.columns:
                city_avg = (
                    df_raw.groupby('city')['price']
                    .mean().sort_values(ascending=False).head(10)
                    .reset_index()
                )
                fig_city = px.bar(
                    city_avg, x='price', y='city', orientation='h',
                    title='Средняя цена по городам',
                    labels={'price': 'Средняя цена (₺)', 'city': ''},
                    color='price', color_continuous_scale='Greens',
                )
                fig_city.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20),
                                       showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig_city, use_container_width=True)

        st.markdown("---")

        st.subheader("Распределение цен по категориям")

        box_col1, box_col2 = st.columns(2)

        with box_col1:
            if 'room_count' in df_raw.columns:
                top_rooms = df_raw['room_count'].value_counts().head(8).index.tolist()
                box_data = df_raw[
                    df_raw['room_count'].isin(top_rooms)
                    & (df_raw['price'] <= df_raw['price'].quantile(0.99))
                ]
                fig_box = px.box(
                    box_data, x='room_count', y='price',
                    title='Цена vs количество комнат',
                    labels={'room_count': 'Планировка', 'price': 'Цена (₺)'},
                    color='room_count',
                )
                fig_box.update_layout(height=400, showlegend=False,
                                      margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_box, use_container_width=True)

        with box_col2:
            if 'building_age' in df_raw.columns:
                top_ages = df_raw['building_age'].value_counts().head(8).index.tolist()
                box_age = df_raw[
                    df_raw['building_age'].isin(top_ages)
                    & (df_raw['price'] <= df_raw['price'].quantile(0.99))
                ]
                fig_box_age = px.box(
                    box_age, x='building_age', y='price',
                    title='Цена vs возраст здания',
                    labels={'building_age': 'Возраст', 'price': 'Цена (₺)'},
                    color='building_age',
                )
                fig_box_age.update_layout(height=400, showlegend=False,
                                          margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_box_age, use_container_width=True)

        st.markdown("---")

        st.subheader("Интерактивное исследование")
        st.caption("Выберите признак для оси X, город и тип недвижимости — график обновится.")

        filter_col1, filter_col2, filter_col3 = st.columns(3)

        with filter_col1:
            x_options = [c for c in ['total_area', 'floor_num', 'floors_total_num', 'rooms_num']
                         if c in df_raw.columns]
            scatter_x = st.selectbox("Признак по оси X:", x_options if x_options else ['price'])

        with filter_col2:
            if 'city' in df_raw.columns:
                city_options = ['Все города'] + df_raw['city'].value_counts().head(15).index.tolist()
                selected_city = st.selectbox("Город:", city_options)
            else:
                selected_city = 'Все города'

        with filter_col3:
            if 'sub_type' in df_raw.columns:
                type_options = ['Все типы'] + df_raw['sub_type'].value_counts().head(10).index.tolist()
                selected_type = st.selectbox("Тип недвижимости:", type_options)
            else:
                selected_type = 'Все типы'

        filtered = df_raw.copy()
        if selected_city != 'Все города' and 'city' in filtered.columns:
            filtered = filtered[filtered['city'] == selected_city]
        if selected_type != 'Все типы' and 'sub_type' in filtered.columns:
            filtered = filtered[filtered['sub_type'] == selected_type]

        if len(filtered) > 5000:
            filtered = filtered.sample(5000, random_state=42)

        if len(filtered) > 0 and scatter_x in filtered.columns:
            hover_cols = [c for c in ['sub_type', 'city', 'room_count']
                          if c in filtered.columns]
            fig_scatter = px.scatter(
                filtered, x=scatter_x, y='price',
                color='sub_type' if 'sub_type' in filtered.columns else None,
                hover_data=hover_cols,
                title=f'Зависимость цены от «{scatter_x}»'
                      f' ({selected_city}, {selected_type})',
                labels={'price': 'Цена (₺)', scatter_x: scatter_x},
                log_y=True,
                opacity=0.5,
            )
            fig_scatter.update_layout(height=500, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption(f"Показано объектов: **{len(filtered):,}** "
                       f"(лог-шкала по оси Y для наглядности)")
        else:
            st.info("Нет данных для выбранных фильтров.")

        st.markdown("---")

        st.subheader("Тепловая карта корреляций")
        st.caption("Взаимосвязи между числовыми признаками. Чем насыщеннее цвет — тем сильнее связь.")

        num_cols = [c for c in ['price', 'total_area', 'floor_num', 'floors_total_num',
                                'rooms_num', 'tom'] if c in df_raw.columns]
        if len(num_cols) >= 2:
            corr_matrix = df_raw[num_cols].corr().round(2)
            fig_heat = px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                title='Корреляция между числовыми признаками',
                aspect='auto',
            )
            fig_heat.update_layout(height=550, margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("---")

        st.subheader("Сравнение моделей")
        st.caption("Метрики качества моделей на тестовой выборке.")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² лучшей модели", f"{predictor.info['r2']:.4f}")
        with col2:
            st.metric("MAE", f"{predictor.info['mae']:,.0f} ₺")
        with col3:
            st.metric("Обучающая выборка", f"{predictor.info['train_size']:,}")
        with col4:
            st.metric("Признаков", len(predictor.feature_columns))

        comparison_df = pd.DataFrame({
            'Модель': ['Ridge', 'XGBoost', 'Random Forest (выбранная)'],
            'R²':     [0.9541, 0.9655, 0.9669],
            'MAE ₺':  [131_710, 112_192, 104_329],
            'RMSE ₺': [411_230, 357_621, 337_667],
            'MAPE %': [39.24, 32.22, 30.88],
        })
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        comp_melted = comparison_df.melt(
            id_vars='Модель',
            value_vars=['R²'],
            var_name='Метрика', value_name='Значение',
        )
        fig_compare = px.bar(
            comp_melted,
            x='Модель', y='Значение', color='Модель',
            title='Сравнение моделей по R²',
            text='Значение',
            color_discrete_sequence=['#3498db', '#e74c3c', '#2ecc71'],
        )
        fig_compare.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig_compare.update_layout(height=400, showlegend=False,
                                  margin=dict(l=20, r=20, t=60, b=20),
                                  yaxis_range=[0.9, 1.0])
        st.plotly_chart(fig_compare, use_container_width=True)


# Вкладка 3: Справка
with tab_help:
    st.header("О приложении")

    st.markdown("""
    ### Что делает приложение

    Приложение прогнозирует стоимость объекта недвижимости в Турции на основе
    его характеристик. Модель машинного обучения (Random Forest) обучена на
    исторических данных объявлений о продаже и аренде недвижимости за 2018–2019 годы.

    ### Как пользоваться

    1. **Вкладка «Прогноз»** — заполните параметры объекта в левой панели,
       нажмите «Рассчитать стоимость». Приложение покажет предсказанную цену
       в турецких лирах, а также в долларах и евро, и доверительный интервал.
    2. **Вкладка «Дашборд»** — интерактивные графики по датасету: распределения
       цен и площадей, средние цены по категориям, тепловая карта корреляций,
       исследование зависимостей с фильтрами и сравнение моделей.
    3. **Вкладка «Справка»** — то, что вы сейчас читаете.
    """)

    st.markdown("---")

    st.subheader("Описание входных полей")

    st.markdown("""
    | Поле | Единица измерения | Описание |
    |---|---|---|
    | **Тип сделки** | — | Продажа или долгосрочная аренда. Посуточная аренда не поддерживается. |
    | **Площадь** | м² | Общая площадь объекта. Допустимый диапазон: 10–500 м². |
    | **Комнат** | N+M | Планировка: N — жилые комнаты, M — гостиная. «2+1» — 2 спальни + гостиная. |
    | **Этаж** | этаж | Этаж расположения объекта. 0 — цоколь. |
    | **Этажей в доме** | этаж | Общая этажность здания. |
    | **Тип недвижимости** | — | Квартира, вилла, резиденция, дача, отдельный дом и др. |
    | **Город** | — | Город в Турции. Модель различает 10 крупнейших городов. |
    | **Отопление** | — | Тип отопительной системы (газовый котёл, центральное, кондиционер и т. д.). |
    | **Возраст здания** | лет | Возраст дома от года постройки. «0» — новостройка. |
    """)

    st.markdown("---")

    st.subheader("О модели")

    st.markdown(f"""
    **Алгоритм:** {predictor.info['name']} ({predictor.info['algorithm']})

    **Метрики качества на тестовой выборке:**

    - **R² = {predictor.info['r2']:.4f}** — модель объясняет {predictor.info['r2']*100:.1f}%
      дисперсии логарифма цены
    - **MAE = {predictor.info['mae']:,} ₺** — средняя абсолютная ошибка прогноза
    - **RMSE = {predictor.info['rmse']:,} ₺** — ошибка с усиленным штрафом за крупные промахи
    - **MAPE = {predictor.info['mape']:.2f}%** — средняя относительная ошибка

    **Как модель получила эти метрики:**

    1. Данные очищены: удалены дубликаты, некорректные цены (≤ 0),
       выбросы по площади и цене ограничены 99-м перцентилем.
    2. Целевая переменная (`price`) логарифмирована (`log1p`) для
       нормализации распределения.
    3. Категориальные признаки закодированы One-Hot Encoding,
       непрерывные — стандартизированы.
    4. Random Forest обучен на 199 767 объектах (80% данных),
       протестирован на 49 942 объектах (20%).

    **Обучающая выборка:** {predictor.info['train_size']:,} объектов
    **Признаков в модели:** {len(predictor.feature_columns)}
    **Период данных:** {predictor.info['period']}
    """)

    st.markdown("---")

    st.subheader("Ограничения")

    st.markdown("""
    - **Период данных** — 2018–2019 гг. Модель не учитывает инфляцию,
      изменения рынка и курсов валют после этого периода. Для
      актуальных прогнозов требуется переобучение на свежих данных.
    - **Районы не учитываются.** Модель различает города, но не различает
      районы внутри города. Для Стамбула это особенно существенно —
      цены в Beşiktaş и Esenyurt могут различаться в 3–5 раз.
    - **Посуточная аренда не поддерживается.** Обучающая выборка
      содержит её лишь частично, модель не может дать осмысленный прогноз.
    - **Элитный сегмент.** Для объектов площадью более 500 м²
      прогноз может иметь повышенную ошибку — таких объектов мало в данных.
    - **Точность ±30%.** Типичная относительная ошибка (MAPE) составляет
      около 31%. Используйте доверительный интервал, а не точечное значение.
    - **Не является оценкой для сделки.** Приложение демонстрирует
      возможности ML-модели, а не заменяет профессиональную оценку
      недвижимости.
    """)

    st.markdown("---")

    st.subheader("Техническая информация")

    st.markdown("""
    **Стек:** Python 3.13 · Streamlit · Plotly · scikit-learn · pandas · numpy

    **Воспроизводимость:** все random_state зафиксированы (`42`),
    модель обучена на фиксированном train/test-разбиении (80/20).
    """)

    st.markdown("**Архитектура приложения:**")
    st.code(
        """app/
├── app.py                    # точка входа, UI (Streamlit)
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # загрузка модели, признаков, метаданных
│   ├── preprocessing.py      # валидация и кодирование ввода
│   └── predictor.py          # фасад для прогноза
├── models/
│   ├── best_model.pkl        # обученный Random Forest
│   └── feature_columns.pkl   # список признаков
└── requirements.txt""",
        language="text",
    )

    st.markdown("""
    **Авторы:** Хайретдинов Тимур и Тагай Кирилл  
    **Версия:** 1.0  
    **Дата:** 15.09.2026
    """)