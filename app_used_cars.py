import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import datetime as dt



# Configuración
st.set_page_config(page_title="Estimador de Precios de Autos Usados", layout="wide")
st.title("🚗💲 Estimador de Precios | Carros Usados")
shap.initjs()

# Cargar pipeline y dataset
pipeline = joblib.load("xgb_car_price_pipeline.pkl")
data = joblib.load("df_filtered.pkl")
features = data.columns

numerical_columns = ['model_year', 'milage',"engine_size_y"]


with st.expander("🧮 Estimación de Precio"):
    # Collect inputs
    selected_brand = st.selectbox("Marca:", options=sorted(data["brand"].unique()))
    
    model_options = sorted(data[data["brand"] == selected_brand]["model"].unique())
    selected_model = st.selectbox("Modelo:", options=model_options)

    # Get the categorized model
    categorized_model = data[data["model"] == selected_model]["categorized_model"].values[0]

    selected_fuel_type = st.selectbox("Tipo de Combustible:", options=sorted(data["fuel_type"].unique()))
    selected_transmission_type = st.selectbox("Tipo de Transmisión:", options=sorted(data["transmission_type"].unique()))
    selected_cat_ext_col = st.selectbox("Color Exterior:", options=sorted(data["cat_ext_col"].unique()))

    selected_ext_col_metal = st.radio("¿El color tiene acabado metalizado?", options=["Sí", "No"])
    ext_col_metal_mapping = {"Sí": 1, "No": 0}
    selected_ext_col_metal = ext_col_metal_mapping[selected_ext_col_metal]

    selected_cat_int_col = st.selectbox("Color Interior:", options=sorted(data["cat_int_col"].unique()))

    selected_accident = st.radio("¿Ha tenido accidentes?", options=["Sí", "No"])
    accident_mapping = {"Sí": "At least 1 accident or damage reported", "No": "None reported"}
    selected_accident = accident_mapping[selected_accident]

    selected_clean_title = st.radio("¿Cuenta con título limpio (Documentación en regla)?", options=["Sí", "No"])
    clean_title_mapping = {"Sí": 1, "No": 0}
    selected_clean_title = clean_title_mapping[selected_clean_title]


    # Numerical Inputs (Sliders)
    cols = st.columns(2)
    numerical_inputs = {}
    for i, col in enumerate(numerical_columns):
        min_val = float(data[col].min())
        max_val = float(data[col].max())
        mean_val = float(data[col].mean())
        if min_val == max_val:
            max_val += 1.0
        step_size = 0.1 if col == "engine_size_y" else 1.0
        with cols[i % 2]:  
            numerical_inputs[col] = st.slider(
                label=f"{col.replace('_', ' ').capitalize()}",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=step_size
            )

# Create DataFrame
input_df = pd.DataFrame([{
    "brand": selected_brand,
    "model_year": numerical_inputs["model_year"],
    "milage": numerical_inputs["milage"],
    "fuel_type": selected_fuel_type,
    "accident": selected_accident,
    "clean_title": selected_clean_title,
    "categorized_model": categorized_model,
    "transmission_type": selected_transmission_type,
    "Luxury_Brand": 0,  # Placeholder (to be calculated)
    "cat_ext_col": selected_cat_ext_col,
    "ext_col_metal": selected_ext_col_metal,
    "cat_int_col": selected_cat_int_col,
    "milage_year": 0,  # Placeholder (to be calculated)
    "engine_size_y": numerical_inputs["engine_size_y"]
}])

# Compute additional features
current_year = dt.date.today().year
input_df["milage_year"] = input_df["milage"] / (current_year - input_df["model_year"])

luxury_brands = [
    "Bugatti", "Rolls-Royce", "Lamborghini", "Ferrari", "McLaren",
    "Maserati", "Bentley", "Aston", "Lucid", "Rivian", "Porsche", "Maybach"
]
input_df["Luxury_Brand"] = input_df["brand"].apply(lambda x: 1 if x in luxury_brands else 0)


if st.button("🔍 Predecir"):
        try:
            # Predict using the model
            pred = pipeline.predict(input_df)[0]
            formatted_price = "💲 El precio estimado es **${:,.2f}**".format(pred)
            st.success(formatted_price)
        except Exception as e:
            st.error(f"❌ Error en la predicción: {e}")
