import pandas as pd
import itertools
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector as selector
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Solicitar ruta del archivo ---
archivo = input("Ingresa la ruta completa del archivo CSV de entrada: ").strip()
if not os.path.isfile(archivo):
    print(f"Archivo no encontrado en la ruta especificada: {archivo}")
    exit(1)

directorio_salida = os.path.dirname(archivo)

# --- Leer datos originales ---
df = pd.read_csv(archivo, skiprows=1)

# Limpiar columna de precio
df['Price;;;'] = df['Price;;;'].astype(str).str.replace(';;;', '', regex=False)
df.rename(columns={'Price;;;': 'price'}, inplace=True)
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Eliminar nulos y duplicados
df_clean = df.dropna().drop_duplicates()

# Filtrar outliers del precio
q1 = df_clean['price'].quantile(0.25)
q3 = df_clean['price'].quantile(0.75)
iqr = q3 - q1
df_clean = df_clean[(df_clean['price'] >= q1 - 1.5 * iqr) & (df_clean['price'] <= q3 + 1.5 * iqr)]

# Variables y target
X = df_clean.drop(columns=['price'])
y = df_clean['price']

# Definir columnas categóricas para codificación
cat_cols = ['Brand', 'Category', 'Color', 'Size', 'Material']

# Pipeline con OneHotEncoder y modelo
print("\n¿Qué modelo deseas usar?")
print("0 - LinearRegression")
print("1 - RandomForest")
print("2 - XGBoost")

modelo_opcion = input("Escribe 0 o 1 o 2: ").strip()

if modelo_opcion == '1':
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    nombre_modelo = 'RandomForest'
elif modelo_opcion == '2':
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='rmse'
    )
    nombre_modelo = 'XGBoost'
else:
    print("Opción default. Usando LinearRegression por defecto.")
    model = LinearRegression()
    nombre_modelo = 'LinearRegression'

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), selector(dtype_include=np.number))
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preproc', preprocessor),
    ('model', model)
])

# Entrenar modelo completo
pipeline.fit(X, y)

# Guardar modelo entrenado
ruta_modelo = os.path.join(directorio_salida, f'modelo_precio_{nombre_modelo}.pkl')
joblib.dump(pipeline, ruta_modelo)
print(f"Modelo {nombre_modelo} entrenado y guardado en {ruta_modelo}.")

# --- Generar todas las combinaciones posibles ---
brands = df_clean['Brand'].unique()
categories = df_clean['Category'].unique()
colors = df_clean['Color'].unique()
sizes = df_clean['Size'].unique()
materials = df_clean['Material'].unique()

combinaciones = list(itertools.product(brands, categories, colors, sizes, materials))
df_combinaciones = pd.DataFrame(combinaciones, columns=['Brand', 'Category', 'Color', 'Size', 'Material'])

# Añadir precio original si existe
df_precio_original = df_clean[['Brand', 'Category', 'Color', 'Size', 'Material', 'price']]
df_combinaciones = df_combinaciones.merge(df_precio_original,
                                          on=['Brand', 'Category', 'Color', 'Size', 'Material'],
                                          how='left')
df_combinaciones.rename(columns={'price': 'Precio_Original'}, inplace=True)

# Preparar para predicción (sin la columna Precio_Original)
df_combinaciones_sin_precio = df_combinaciones.drop(columns=['Precio_Original'])

# Predecir precios
predicciones = pipeline.predict(df_combinaciones_sin_precio)
df_combinaciones['Precio_Predicho'] = predicciones

# Mostrar primeras 20 filas resultado
print("\n20 primeras filas de las predicciones por segmento:")
print(df_combinaciones.head(20))

# Calcular y mostrar MSE sobre los datos con precio original conocido
df_analisis = df_combinaciones.dropna(subset=['Precio_Original'])
mse_analisis = mean_squared_error(df_analisis['Precio_Original'], df_analisis['Precio_Predicho'])
print(f"\nMSE en filas con precio original conocido: {mse_analisis:.2f}")

# Preguntar si quiere ver solo filas con precio original
ver_filtrado = input("\n¿Quieres ver sólo filas con precio original conocido? (s/n): ").strip().lower()
if ver_filtrado == 's':
    df_filtrado = df_analisis
    print("\n20 primeras filas con precio original conocido:")
    print(df_filtrado.head(20))

    # Guardar parquet filtrado
    output_parquet_filtrado = os.path.join(directorio_salida, 'predicciones_por_segmento_filtrado.parquet')
    df_filtrado.to_parquet(output_parquet_filtrado, index=False)
    print(f"Guardado archivo parquet filtrado: {output_parquet_filtrado}")

    # Guardar JSON filtrado opcional
    guardar_json_filtrado = input("\n¿Deseas guardar también el archivo JSON sólo con filas filtradas? (s/n): ").strip().lower()
    if guardar_json_filtrado == 's':
        output_json_filtrado = os.path.join(directorio_salida, 'predicciones_por_segmento_filtrado.json')
        df_filtrado.to_json(output_json_filtrado, orient='records', indent=2)
        print(f"Guardado en JSON filtrado: {output_json_filtrado}")

# Guardar parquet general
output_parquet = os.path.join(directorio_salida, 'predicciones_por_segmento.parquet')
df_combinaciones.to_parquet(output_parquet, index=False)
print(f"Guardado archivo parquet: {output_parquet}")

# Guardar JSON general opcional
guardar_json = input("\n¿Deseas guardar también en JSON el archivo completo? (s/n): ").strip().lower()
if guardar_json == 's':
    output_json = os.path.join(directorio_salida, 'predicciones_por_segmento.json')
    df_combinaciones.to_json(output_json, orient='records', indent=2)
    print(f"Guardado en JSON: {output_json}")

# --- Análisis visual de predicciones ---
generar_grafica = input("\n¿Deseas generar gráficas de análisis? (s/n): ").strip().lower()

if generar_grafica != 's':
    print("Análisis finalizado sin visualizaciones.")
    exit()

if not df_analisis.empty:
    # Gráfico 1: Precio Original vs Precio Predicho
    plt.figure(figsize=(8,6))
    sns.scatterplot(x='Precio_Original', y='Precio_Predicho', data=df_analisis, alpha=0.6)
    min_val = min(df_analisis['Precio_Original'].min(), df_analisis['Precio_Predicho'].min())
    max_val = max(df_analisis['Precio_Original'].max(), df_analisis['Precio_Predicho'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (y = x)')
    plt.title('Comparación Precio Original vs Precio Predicho')
    plt.xlabel('Precio Original')
    plt.ylabel('Precio Predicho')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Gráfico 2: Errores de predicción
    residuals = df_analisis['Precio_Original'] - df_analisis['Precio_Predicho']
    plt.figure(figsize=(8,6))
    sns.histplot(residuals, bins=30, kde=True)
    plt.title('Distribución de errores (Precio Original - Precio Predicho)')
    plt.xlabel('Error')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.show()
else:
    print("\nNo hay suficientes datos con Precio Original para graficar análisis.")