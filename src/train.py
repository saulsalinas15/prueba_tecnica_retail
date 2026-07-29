import pandas as pd
import numpy as np
import lightgbm as lgb
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_model(df_clean):
    """
    Recibe el DataFrame limpio, separa el set de validación (últimos 30 días),
    convierte categóricas, entrena el LightGBM optimizado y registra en MLflow.
    """
    print("[INFO] Iniciando proceso de modelado")

    # ==========================================
    # 1. PARTICIONADO TEMPORAL (Split)
    # ==========================================
    # Regla estricta: Últimos 30 días para validación (Febrero 2024)
    val_cutoff = '2024-01-30'
    train_df = df_clean[df_clean['date'] <= val_cutoff].copy()
    val_df = df_clean[df_clean['date'] > val_cutoff].copy()

    print(f"[INFO] Entrenando con {len(train_df)} registros. Validando con {len(val_df)} registros.")

    # ==========================================
    # 2. PREPARACIÓN DE MATRICES (X, y)
    # ==========================================
    cols_no_features = ['date', 'units_sold']
    
    # Conversión dinámica de categóricas (textos a category para LightGBM)
    object_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    for col in object_cols:
        train_df[col] = train_df[col].astype('category')
        val_df[col] = val_df[col].astype('category')

    features = [col for col in train_df.columns if col not in cols_no_features]
    
    X_train, y_train = train_df[features], train_df['units_sold']
    X_val, y_val = val_df[features], val_df['units_sold']

    # ==========================================
    # 3. ENTRENAMIENTO (Modelo Optimizado)
    # ==========================================
    # Usamos los hiperparámetros ganadores en el experimento
    best_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'max_depth': 7,
        'n_estimators': 150,
        'num_leaves': 31,
        'random_state': 42,
        'verbose': -1
    }

    print("[INFO] Entrenando LightGBM con hiperparámetros optimizados")
    model_lgb = lgb.LGBMRegressor(**best_params)
    model_lgb.fit(X_train, y_train)

    # ==========================================
    # 4. EVALUACIÓN Y REGLAS DE NEGOCIO
    # ==========================================
    preds = model_lgb.predict(X_val)
    # Regla: No existen ventas negativas
    preds = np.clip(preds, a_min=0, a_max=None)

    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mae = mean_absolute_error(y_val, preds)

    # ==========================================
    # 5. REGISTRO EN MLFLOW
    # ==========================================
    print("[INFO] Registrando experimento en MLflow...")
    mlflow.set_experiment("Retail_Demand_Forecasting_PROD")
    
    with mlflow.start_run(run_name="LightGBM_Tuned_Production"):
        mlflow.log_params(best_params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        
        # También podemos guardar el modelo mismo
        mlflow.lightgbm.log_model(model_lgb, name="model")

    print("\n" + "="*50)
    print("🏆 ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("="*50)
    print(f"RMSE Final: {rmse:.4f}")
    print(f"MAE Final : {mae:.4f}")
    print("="*50)

    return model_lgb, rmse, mae

# Nota Arquitectónica: En un sistema de producción maduro, estos hiperparámetros 
# no estarían hardcodeados, sino que se inyectarían dinámicamente desde un 
# archivo 'config.yaml' o un Parameter Store tras cada ciclo de Continuous Training.
#    best_params = {
#        'objective': 'regression',
#        ...