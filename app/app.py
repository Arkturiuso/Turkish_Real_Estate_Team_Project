import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import altair as alt

from app.src.predictor import get_predictor

# Настройка страницы
st.set_page_config(page_title="Прогноз цен на недвижимость", layout="wide")
st.title("Прогнозирование стоимости недвижимости в Турции")

# Инициализация предиктора
predictor = get_predictor()

# Боковая панель с параметрами
st.sidebar.header("Параметры объекта")
user_input = {
    'total_area': st.sidebar.number_input("Площадь (м²)", 10.0, 1000.0, 100.0, 5.0),
    'room_count': st.sidebar.selectbox("Комнат", ['1+0', '1+1', '2+1', '3+1', '3+2', '4+2', '5+2'], index=2),
    'floor_num': st.sidebar.number_input("Этаж", 0, 50, 3),
    'floors_total_num': st.sidebar.number_input("Этажей в доме", 1, 100, 10),
    'sub_type': st.sidebar.selectbox("Тип", ['Daire', 'Villa', 'Müstakil Ev', 'Rezidans', 'Yazlık', 'Другие']),
    'city': st.sidebar.selectbox("Город", ['İstanbul', 'Ankara', 'İzmir', 'Antalya', 'Aydın', 'Muğla', 'Mersin', 'Другие']),
    'heating_type': st.sidebar.selectbox("Отопление", ['Kombi (Doğalgaz)', 'Merkezi Sistem', 'Klima', 'Yerden Isıtma', 'Soba (Kömür)', 'Другие']),
    'building_age': st.sidebar.selectbox("Возраст здания", ['0', '1', '2', '3', '4', '5', '6-10 arası', '11-15 arası', '16+', 'Другие']),
}

# Вкладки приложения
tab_form, tab_dashboard, tab_help = st.tabs(["Прогноз", "Дашборд", "Справка"])

# Вкладка 1: Прогноз
with tab_form:
    st.header("Расчет стоимости объекта")
    if st.button("Рассчитать стоимость", type="primary", use_container_width=True):
        try:
            result = predictor.predict(user_input)
            
            col1, col2, col3 = st.columns(3)
            with col1: 
                st.metric("Предсказанная цена", f"{result['price']:,.0f} ₺")
            with col2: 
                st.metric("В долларах", f"${result['price']/30:,.0f}")
            with col3: 
                st.metric("В евро", f"€{result['price']/33:,.0f}")
            
            st.info(f"**Доверительный интервал (±{result['mape_pct']}%):**\n"
                    f"от {result['lower_bound']:,.0f} ₺ до {result['upper_bound']:,.0f} ₺")
        except ValueError as e:
            st.error(f"Ошибка валидации: {e}")
        except Exception as e:
            st.error(f"Произошла ошибка при расчете: {e}")

# Вкладка 2: Дашборд
with tab_dashboard:
    st.header("Дашборд исследования данных")
    
    try:
        df_raw = pd.read_csv('cleaned_real_estate.csv')
    except FileNotFoundError:
        st.warning("Файл `cleaned_real_estate.csv` не найден. Убедитесь, что он находится в той же папке.")
        df_raw = None
    
    if df_raw is not None:
        st.subheader("Общая статистика по датасету")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Всего объектов", f"{len(df_raw):,}")
        with col2: st.metric("Средняя цена", f"{df_raw['price'].mean():,.0f} ₺")
        with col3: st.metric("Медианная цена", f"{df_raw['price'].median():,.0f} ₺")
        
        area_col = 'total_area' if 'total_area' in df_raw.columns else 'rooms'
        with col4: 
            if area_col in df_raw.columns:
                st.metric("Средняя площадь", f"{df_raw[area_col].mean():.1f} м²")
            else:
                st.metric("Средняя площадь", "Н/Д")
        
        st.markdown("---")
        
        st.subheader("Распределения признаков")
        dist_col1, dist_col2 = st.columns(2)
        
        with dist_col1:
            st.markdown("#### Распределение цен")
            st.altair_chart(
                alt.Chart(df_raw).mark_bar().encode(
                    x=alt.X('price:Q', bin=alt.Bin(maxbins=30), title='Цена (₺)'),
                    y=alt.Y('count()', title='Количество'),
                    tooltip=['count()']
                ).properties(height=300),
                use_container_width=True
            )
        
        with dist_col2:
            st.markdown(f"#### Распределение ({area_col})")
            if area_col in df_raw.columns:
                st.altair_chart(
                    alt.Chart(df_raw).mark_bar().encode(
                        x=alt.X(f'{area_col}:Q', bin=alt.Bin(maxbins=30), title=area_col),
                        y=alt.Y('count()', title='Количество'),
                        tooltip=['count()']
                    ).properties(height=300),
                    use_container_width=True
                )
            else:
                st.info(f"Столбец '{area_col}' был удален на этапе предобработки.")
        
        st.markdown("---")
        
        st.subheader("Средние цены по категориям")
        cat_col1, cat_col2 = st.columns(2)
        
        with cat_col1:
            st.markdown("#### По типу недвижимости")
            if 'sub_type' in df_raw.columns:
                type_avg = df_raw.groupby('sub_type')['price'].mean().sort_values(ascending=False).head(10)
                fig_type = px.bar(
                    x=type_avg.values, y=type_avg.index, orientation='h',
                    title='Средняя цена по типам', labels={'x': 'Средняя цена (₺)', 'y': 'Тип'}
                )
                st.plotly_chart(fig_type, use_container_width=True)
        
        with cat_col2:
            st.markdown("#### По возрасту здания")
            if 'building_age' in df_raw.columns:
                age_avg = df_raw.groupby('building_age')['price'].mean().sort_values(ascending=False).head(10)
                fig_age = px.bar(
                    x=age_avg.values, y=age_avg.index, orientation='h',
                    title='Средняя цена по возрасту', labels={'x': 'Средняя цена (₺)', 'y': 'Возраст'}
                )
                st.plotly_chart(fig_age, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("Интерактивное исследование")
        available_scatter_cols = [c for c in ['total_area', 'floor_num', 'floors_total_num'] if c in df_raw.columns]
        if not available_scatter_cols:
            available_scatter_cols = ['price']
            
        scatter_x = st.selectbox("Выберите признак для оси X:", available_scatter_cols)
        
        scatter_col1, scatter_col2 = st.columns(2)
        with scatter_col1:
            tooltip_cols = [c for c in [scatter_x, 'price', 'sub_type'] if c in df_raw.columns]
            # Создаем график
            fig_scatter = px.scatter(
                df_raw, x=scatter_x, y='price', 
                color='sub_type' if 'sub_type' in df_raw.columns else None,
                hover_data=tooltip_cols, title=f'Зависимость цены от {scatter_x}', log_y=True
            )
            # Убираем width='stretch' отсюда, он здесь недопустим
            fig_scatter.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
            # Растягивание управляется здесь:
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with scatter_col2:
            if 'city' in df_raw.columns:
                tooltip_cols2 = [c for c in [scatter_x, 'price', 'city'] if c in df_raw.columns]
                fig_scatter_city = px.scatter(
                    df_raw, x=scatter_x, y='price', color='city',
                    hover_data=tooltip_cols2, title=f'Цена vs {scatter_x} (по городам)', log_y=True
                )
                fig_scatter_city.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_scatter_city, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Сравнение моделей")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R²", f"{predictor.info['r2']:.4f}", delta=f"{predictor.info['r2']*100:.1f}% объяснённой дисперсии")
    with col2:
        st.metric("MAPE", f"{predictor.info['mape']:.2f}%", delta="средняя ошибка", delta_color="inverse")
    with col3:
        st.metric("Обучающая выборка", f"{predictor.info['train_size']:,}")
    with col4:
        st.metric("Признаков", len(predictor.feature_columns))
    
    st.markdown("#### Сравнение всех моделей")
    comparison_df = pd.DataFrame({
        'Модель': ['Ridge', 'Random Forest', 'XGBoost (выбранная)'],
        'R²': [0.9314, 0.9444, 0.9470],
        'MAE': [0.465, 0.412, 0.402],
        'MAPE': [4.58, 3.90, 3.81],
        'RMSE': [0.678, 0.610, 0.596]
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    fig_compare = px.bar(
        comparison_df, x='Модель', y=['R²', 'MAPE'], title='Сравнение моделей по метрикам',
        barmode='group', labels={'value': 'Значение', 'variable': 'Метрика'}
    )
    st.plotly_chart(fig_compare, use_container_width=True)

# Вкладка 3: Справка
with tab_help:
    st.header("О модели и ограничениях")
    st.markdown(f"""
    **Алгоритм:** {predictor.info['name']} (градиентный бустинг)
    
    **Качество на тестовой выборке:**
    - **R²** = {predictor.info['r2']} (объясняет {predictor.info['r2']*100:.0f}% дисперсии)
    - **MAPE** = {predictor.info['mape']}% (средняя ошибка прогноза ~4%)
    
    **Ограничения:**
    - Модель обучена на данных 2018–2019 годов.
    - Прогноз для элитных объектов (площадь > 500 м²) может иметь повышенную ошибку.
    - Рекомендуется использовать указанный доверительный интервал при интерпретации результатов.
    """)