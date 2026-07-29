import warnings
import os
from src.data_prep import load_and_prep_data
from src.train import train_model

os.environ["GIT_PYTHON_REFRESH"] = "quiet"
warnings.filterwarnings('ignore')

def run_pipeline():
    print("="*60)
    print("🚀 INICIANDO PIPELINE DE ML PARA WALMART")
    print("="*60)

    # 1. Rutas de los archivos crudos
    path_trans = "data/transactions.csv"
    path_stores = "data/stores.csv"
    path_calendar = "data/calendar.csv"

    # 2. Ejecutar Fase de Preparación de Datos
    print("\n>>> FASE 1: Preparación de Datos")
    df_clean = load_and_prep_data(path_trans, path_stores, path_calendar)

    # 3. Ejecutar Fase de Entrenamiento
    print("\n>>> FASE 2: Entrenamiento y Registro en MLflow")
    model, rmse, mae = train_model(df_clean)

    print("\n" + "="*60)
    print("✅ PIPELINE EJECUTADO CON ÉXITO")
    print("="*60)

if __name__ == "__main__":
    run_pipeline()