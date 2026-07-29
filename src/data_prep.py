import pandas as pd
import numpy as np

def load_and_prep_data(path_trans, path_stores, path_calendar):
    """
    Carga las transacciones, cruza con tiendas y calendario,
    imputa nulos, elimina variables con Data Leakage y genera variables temporales.
    """
    print("[INFO] Cargando y cruzando archivos de datos")
    df_trans = pd.read_csv(path_trans)
    df_stores = pd.read_csv(path_stores)
    df_calendar = pd.read_csv(path_calendar)

    # ==========================================
    # 0. FORMATO DE FECHAS Y JOINS
    # ==========================================
    df_trans['date'] = pd.to_datetime(df_trans['date'])
    df_calendar['date'] = pd.to_datetime(df_calendar['date'])

    df = pd.merge(df_trans, df_stores, on='store_id', how='left')
    df = pd.merge(df, df_calendar, on='date', how='left')

    # ==========================================
    # 1. LIMPIEZA Y ORDENAMIENTO
    # ==========================================
    df = df.drop_duplicates()
    
    # Ordenar cronológicamente por tienda y categoría 
    df = df.sort_values(by=['store_id', 'category', 'date']).reset_index(drop=True)

    # ==========================================
    # 2. IMPUTACIÓN DE NULOS
    # ==========================================
    print("[INFO] Imputando nulos por apagones de POS (Interpolación)...")
    columnas_a_imputar = ['units_sold', 'amount_cash', 'avg_ticket']
    for col in columnas_a_imputar:
        if col in df.columns:
            df[col] = df.groupby(['store_id', 'category'])[col].transform(
                lambda x: x.interpolate(method='linear', limit_direction='both')
            )
            df[col] = df[col].fillna(0)

    # ==========================================
    # 3. PREVENCIÓN DE DATA LEAKAGE
    # ==========================================
    # Eliminamos 'replenishment_signal' y variables financieras simultáneas
    columnas_tramposas = ['replenishment_signal', 'amount_cash', 'avg_ticket', 'amount_total', 'amount_card']
    cols_to_drop = [c for c in columnas_tramposas if c in df.columns]
    
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"[INFO] Columnas eliminadas por Data Leakage: {cols_to_drop}")

    # ==========================================
    # 4. INGENIERÍA DE VARIABLES (FEATURES)
    # ==========================================
    print("[INFO] Creando variables temporales y de rezago (Lags)...")
    
    # Calendario
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month

    # Lags y Media Móvil
    if 'units_sold' in df.columns:
        grupos = df.groupby(['store_id', 'category'])['units_sold']
        
        df['lag_1'] = grupos.shift(1)
        df['lag_7'] = grupos.shift(7)
        
        df['rolling_mean_7'] = grupos.transform(
            lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
        )

    # Limpiar nulos generados por el shift inicial
    cols_hist = ['lag_1', 'lag_7', 'rolling_mean_7']
    df[cols_hist] = df[cols_hist].fillna(0)

    print(f"[EXITO] Datos preparados. Shape final: {df.shape}")
    return df

# Bloque de prueba local
if __name__ == "__main__":
    # Asegúrate de tener estos archivos en tu carpeta data/
    df_clean = load_and_prep_data(
        path_trans="../data/transactions.csv",
        path_stores="../data/stores.csv",
        path_calendar="../data/calendar.csv"
    )
    print(df_clean.head())