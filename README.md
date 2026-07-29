# Retail Demand Forecasting — Production ML Pipeline

Sistema de respaldo predictivo diseñado para estimar la demanda diaria de mercancía a nivel tienda y categoría en Retail. La solución actúa como un **motor inercial de demanda** que mitiga la pérdida de visibilidad generada por fallas intermitentes en el sistema de punto de venta (POS).

Para una explicación detallada sobre las decisiones de negocio, prevención de *Data Leakage*, rigor estadístico y uso de IA, consulta el documento **[PROCESS.md](./PROCESS.md)**.

---

##  Estructura del Repositorio

```text
retail_forecasting/
├── data/                  # Datos históricos (transactions, stores, calendar)
├── notebooks/
│   └── 01_eda.ipynb       # Análisis exploratorio (EDA) y prototipado inicial
├── src/
│   ├── __init__.py
│   ├── data_prep.py       # Módulo de ingesta, limpieza e ingeniería de variables
│   └── train.py           # Módulo de entrenamiento y registro en MLflow
├── main.py                # Orquestador principal del pipeline de entrenamiento
├── predict.py             # Script de inferencia (simulación de predicción diaria)
├── PROCESS.md             # Documentación metodológica y justificación técnica
├── README.md              # Guía de instalación y uso
├── requirements.txt       # Dependencias del proyecto
└── .gitignore             # Filtros de versionado para Git
```

---

## Guía de Instalación y Ejecución

### 1. Requisitos Previos
Asegúrate de contar con **Python 3.9+** instalado en tu sistema.

### 2. Clonar el Repositorio e Instalar Dependencias
```bash
# Clonar repositorio
git clone <URL_DE_TU_REPOSITORIO_EN_GITHUB>
cd retail_forecasting

# Crear y activar entorno virtual
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Mac/Linux:
source venv/bin/activate

# Instalar librerías requeridas
pip install -r requirements.txt
```

---

## Ejecución del Pipeline

### Opción A: Entrenamiento Completo (`main.py`)
Ejecuta el pipeline end-to-end: procesa los datos crudos, aplica feature engineering, entrena el modelo LightGBM y registra los artefactos y métricas en MLflow.

```bash
python main.py
```

### Opción B: Inferencia / Predicción (`predict.py`)
Simula la generación de pronósticos diarios cargando el modelo campeón registrado en MLflow:

```bash
python predict.py
```

---

## Seguimiento de Experimentos con MLflow

Para inspeccionar las métricas de rendimiento (RMSE, MAE) y los hiperparámetros registrados durante los entrenamientos, abre la interfaz visual de MLflow ejecutando:

```bash
mlflow ui
```
Luego navega a `http://127.0.0.1:5000` en tu navegador web.

---

## 📈 Resultados del Modelo

* **Algoritmo:** LightGBM Regressor
* **Estrategia de Validación:** Split temporal estricto (últimos 30 días)
* **RMSE Final:** ~311.81 unidades
* **MAE Final:** ~204.01 unidades