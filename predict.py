import pandas as pd
import mlflow
import warnings
import os
from src.data_prep import load_and_prep_data

warnings.filterwarnings('ignore')
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

def predict_future_demand():
    """
    Simula el proceso de inferencia diario. Carga los últimos datos,
    descarga el modelo campeón de MLflow y genera las predicciones.
    """
    print("="*60)
    print("🔮 SISTEMA DE PREDICCIÓN DE DEMANDA (INFERENCIA)")
    print("="*60)

    # 1. Cargar datos (Simulando los datos del "día de hoy")
    print("[INFO] Cargando datos recientes...")
    df = load_and_prep_data(
        "data/transactions.csv", 
        "data/stores.csv", 
        "data/calendar.csv"
    )
    
    # Para la simulación, tomaremos solo las últimas 5 filas (las más recientes)
    # y ocultaremos la variable 'units_sold' para fingir que es el futuro
    df_future = df.tail(5).copy()
    X_future = df_future.drop(columns=['date', 'units_sold'])
    
    # 2. Conectar a MLflow y buscar el modelo más reciente
    print("[INFO] Buscando el modelo campeón en MLflow")
    experiment = mlflow.get_experiment_by_name("Retail_Demand_Forecasting_PROD")
    
    if experiment is None:
        print("[ERROR] No se encontró el experimento. Entrena el modelo primero.")
        return
        
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], 
        order_by=["start_time DESC"], 
        max_results=1
    )
    latest_run_id = runs.iloc[0].run_id
    
    # 3. Descargar modelo y predecir
    print(f"[INFO] Modelo cargado exitosamente (Run ID: {latest_run_id})")
    model_uri = f"runs:/{latest_run_id}/model"
    model = mlflow.lightgbm.load_model(model_uri)
    
    # Conversión de categóricas (textos a category para LightGBM)
    object_cols = X_future.select_dtypes(include=['object']).columns.tolist()
    for col in object_cols:
        X_future[col] = X_future[col].astype('category')

    print("[INFO] Generando predicciones...\n")
    predicciones = model.predict(X_future)
    
    # 4. Mostrar resultados
    df_future['prediccion_demanda_unidades'] = predicciones.round(0).astype(int) # Redondeamos a unidades enteras
    
    print("📊 RESULTADOS DEL PRONÓSTICO PARA EL SIGUIENTE CICLO:")
    print("-" * 60)
    print(df_future[['store_id', 'category', 'date', 'prediccion_demanda_unidades']])
    print("-" * 60)

if __name__ == "__main__":
    predict_future_demand()