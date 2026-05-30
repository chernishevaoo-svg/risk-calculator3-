import streamlit as st
import json
import numpy as np
import os

# ==========================================
# ЗАГРУЗКА ПАРАМЕТРОВ МОДЕЛИ
# ==========================================
@st.cache_data
def load_params():
    json_path = 'model_params.json'
    if not os.path.exists(json_path):
        st.error(f"Файл {json_path} не найден! Убедитесь, что он загружен в репозиторий GitHub рядом с app.py.")
        st.stop()
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

params = load_params()

# ==========================================
# НАСТРОЙКА СТРАНИЦЫ
# ==========================================
st.set_page_config(page_title="Калькулятор риска", page_icon="⚕️", layout="wide")

st.title("⚕️ Прогнозирование летального исхода у пациентов с бактериальными менингитами после третьих суток госпитализации")
st.markdown("Введите лабораторные показатели пациента, полученные на третьи сутки госпитализации, для расчета вероятности неблагоприятного исхода.")

# ==========================================
# ФОРМА ВВОДА ДАННЫХ
# ==========================================
feature_names = params['feature_names']
coefficients = params['coefficients']
intercept = params['intercept']
orig_medians = params['orig_medians']
inverted_features = params.get('inverted_features', [])
orig_min_max = params.get('orig_min_max', {})
prob_low = params['prob_low']
prob_high = params['prob_high']

st.write("### Лабораторные и клинические показатели")

# Создаем колонки для компактности
cols = st.columns(3)
user_inputs = {}

for i, feat in enumerate(feature_names):
    col = cols[i % 3]
    
    # Получаем параметры для поля ввода
    default_val = orig_medians[i]
    min_val = orig_min_max.get(feat, [None, None])[0]
    max_val = orig_min_max.get(feat, [None, None])[1]
    
    # Форматируем значение по умолчанию
    if isinstance(default_val, float):
        step = 0.01
        fmt = "%.2f"
    else:
        step = 1.0
        fmt = "%d"

    with col:
        user_inputs[feat] = st.number_input(
            label=f"{feat}",
            min_value=float(min_val) if min_val is not None else None,
            max_value=float(max_val) if max_val is not None else None,
            value=float(default_val),
            step=step,
            format=fmt,
            key=feat
        )

# ==========================================
# РАСЧЕТ ВЕРОЯТНОСТИ
# ==========================================
if st.button("Рассчитать риск", type="primary", use_container_width=True):
    linear_predictor = intercept
    
    for i, feat in enumerate(feature_names):
        val = user_inputs[feat]
        
        # Если признак был инвертирован при обучении (напр, цитоз), инвертируем и ввод
        if feat in inverted_features:
            val_transformed = -val
        else:
            val_transformed = val
            
        linear_predictor += coefficients[i] * val_transformed
    
    # Вычисление вероятности
    probability = 1 / (1 + np.exp(-linear_predictor))
    
    st.divider()
    st.write("### Результат расчета")
    
    # Отображение вероятности
    prob_percent = probability * 100
    st.metric(label="Прогнозируемая вероятность летального исхода", value=f"{prob_percent:.1f}%")
    
    # Определение группы риска
    if probability < prob_low:
        risk_group = "НИЗКИЙ РИСК"
        risk_color = "green"
        risk_emoji = "✅"
        recommendation = "Стабилизация витальных функций. Рассмотреть возможность продолжение наблюдения в инфекционном отделении. Продолжение проводимой антибактериальной терапии. Стандартный лабораторный мониторинг."
    elif probability >= prob_high:
        risk_group = "ВЫСОКИЙ РИСК"
        risk_color = "red"
        risk_emoji = "🔴"
        recommendation = "Продолжить наблюдение в ОРИТ. Рекомендуется смена антибактериальной терапии. Рекомендуется экстренное проведение неровизуализации для исключения осложнений, дополнительного обследования для исключения других очагов инфекционного процесса. Расширение объема интенсивной терапии, применение экстракорпоральных методов лечения. Интенсивный лабораторный мониторинг. Поддержание целевых значений САД"
    else:
        risk_group = "УМЕРЕННЫЙ РИСК"
        risk_color = "orange"
        risk_emoji = "⚠️"
        recommendation = "Продолжить наблюдение в ОРИТ. Рассмотреть вопрос о смене антибактериальной терапии. Интенсивный лабораторный мониторинг. Проведение нейровизуализации для исключения осложнений"
        
    st.markdown(f"<h2 style='text-align: center; color: {risk_color};'>{risk_emoji} {risk_group}</h2>", unsafe_allow_html=True)
    
    st.info(f"**Рекомендации:**\n\n{recommendation}")
    
    # Пороги стратификации
    with st.expander("Пороги стратификации (для врачей)"):
        st.markdown(f"""
        * **Низкий риск:** вероятность < {prob_low*100:.1f}% (Чувствительность 100%)
        * **Умеренный риск:** вероятность от {prob_low*100:.1f}% до {prob_high*100:.1f}%
        * **Высокий риск:** вероятность ≥ {prob_high*100:.1f}% (Высокая специфичность)
        """)
