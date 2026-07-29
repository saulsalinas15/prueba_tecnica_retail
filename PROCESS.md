# Proceso de Desarrollo y Toma de Decisiones (Data Science)

El presente documento detalla el razonamiento analítico, estadístico y técnico detrás de la solución construida para el pronóstico de demanda retail, abordando los puntos críticos del ciclo de vida del Machine Learning (ML).

## 1. Framing del Problema
A través del Análisis Exploratorio de Datos (EDA), se identificó un "dolor" operativo crítico y constante: las fallas sistemáticas en el sistema de punto de venta (POS) generan ceguera transaccional (`NaN`) en el registro diario de ventas (`units_sold`).

**Cuantificación del Problema:**
* **Frecuencia del problema:** En **el 100% de los días analizados (425 de 425 días)** existieron interrupciones del sistema POS.
* **Impacto operativo:** Se pierde diariamente entre el 1% y **hasta un 5.7% de la visibilidad total de las ventas** en un solo día.

**Decisión de Negocio e Impacto en Cadena de Suministro:**
Cuando la regla automática de resurtido corporativo recibe estos valores nulos (`NaN`), la lógica no puede calcular los niveles reales de consumo ni los puntos de reorden. Esto provoca dos cuellos de botella:
1. Paralización del reabastecimiento automático por falta de lectura.
2. Estimaciones erróneas si la lógica tradicional asume ausencia de demanda, congelando el envío de inventario y provocando quiebres de anaquel (*out-of-stock*).

**La Solución:** Se construyó un **Sistema de Respaldo Predictivo** (Motor Inercial de Demanda). Ante la presencia continua de nulos por fallas del POS, este sistema inyecta una estimación estadísticamente sólida de la demanda esperada para mantener el flujo óptimo de inventario. Se descartó el uso de la variable `replenishment_signal` histórica, ya que contiene reglas de emergencia contaminadas por estas mismas fallas.

## 2. Prevención de Data Leakage y Rigor Estadístico
Para garantizar que el modelo sea robusto en un entorno de producción (inferencia), se tomaron medidas estrictas para evitar el filtrado de información (*Data Leakage*):
* **Limpieza de variables simultáneas:** Se eliminaron variables financieras como `amount_cash`, `avg_ticket` y `amount_total`, ya que ocurren en el momento de la venta y no están disponibles en el futuro al momento de emitir un pronóstico.
* **Particionado Temporal:** Tratándose de series de tiempo, se descartó el tradicional `train_test_split` aleatorio. La validación se realizó mediante un corte temporal estricto (últimos 30 días), obligando al modelo a predecir el futuro real.

## 3. Ingeniería de Características (Feature Engineering)
Dado que las fallas del POS inutilizan los datos en tiempo real, el modelo se ancló en variables deterministas (el calendario) e inercia histórica:
1. **Tratamiento de Nulos:** Interpolación lineal/temporal para reconstruir la continuidad de la serie histórica sin alterar la tendencia real de la demanda.
2. **Calendario:** Extracción de estacionalidades fuertes (quincena, fin de semana, mes).
3. **Lags (Rezagos):** Desplazamiento temporal de las ventas en *T-1* y *T-7* para entender el comportamiento inmediato e inter-semanal por tienda y categoría.
4. **Rolling Windows:** Medias móviles de 7 días (excluyendo el día actual) para suavizar el ruido diario y capturar tendencias macro de demanda.

## 4. Selección del Modelo y Análisis de Trade-offs
Para la elección del algoritmo se evaluaron distintas familias de modelos bajo el criterio de impacto en negocio y la restricción temporal del proyecto (48h):

* **vs. Modelos Estadísticos Clásicos (ARIMA / Prophet):** Descartados. Requieren entrenar y mantener un modelo independiente por cada combinación de tienda y categoría (cientos de modelos), lo que no escala eficientemente y no permite compartir patrones globales entre tiendas.
* **vs. Deep Learning (LSTM / Temporal Fusion Transformers):** Descartados para esta fase. Poseen un costo computacional y de ajuste muy alto, además de requerir un volumen de datos sustancialmente mayor para superar a modelos basados en árboles.
* **vs. XGBoost:** Aunque ofrece rendimiento similar, **LightGBM** fue seleccionado por su algoritmo de construcción de árboles basado en histogramas (*leaf-wise growth*), el cual es hasta 10 veces más rápido en entrenamiento y tiene un soporte nativo superior para variables categóricas de retail.
* **Resultado:** LightGBM representa el estándar de la industria logística (validado en competencias como como M5 de Kaggle) al ofrecer el mejor equilibrio entre velocidad, interpretabilidad y precisión.

## 5. Fase de Experimentación, Evaluación y Justificación de Métricas

El desarrollo del modelo siguió un enfoque iterativo y de mejora continua. Para medir el valor real del algoritmo en el conjunto de prueba temporal (últimos 30 días), se estableció un *Baseline* (modelo base sin aprendizaje profundo) y se comparó contra iteraciones sucesivas de LightGBM.

**Evolución del Error y Optimización:**
* **Baseline (Estimación base):** 497.43 unidades
* **LightGBM Base (Out-of-the-box):** 312.50 unidades
* **LightGBM Tuned (Optimizado):** 311.81 unidades

Mediante el ajuste de hiperparámetros (`learning_rate: 0.05, max_depth: 7, n_estimators: 150, num_leaves: 31`), se logró una **mejora total del 37.32% vs el Baseline**. Esto demuestra empíricamente que la inercia histórica capturada por los árboles de decisión aporta un valor predictivo masivo frente a reglas de estimación simples.

**Análisis Estadístico y Operativo de Métricas Finales:**
Para la evaluación final se consolidaron el **MAE** (~204.01) y **RMSE** (~311.81), seleccionados por su estricta alineación con la operación logística:

1. **Interpretación de Negocio (MAE):** En promedio, el pronóstico se desvía por ~204 piezas por día. Este valor proporciona un insumo directo al equipo de logística para calcular un **stock de seguridad** de alrededor de 200 unidades por categoría y tienda, absorbiendo la incertidumbre de la demanda.
2. **Penalización de Errores Graves (RMSE):** En retail, un desvío masivo aislado es mucho más destructivo que múltiples desviaciones pequeñas, ya que provoca vaciados inmediatos de anaquel. La razón RMSE / MAE ~ 1.52 confirma una distribución de errores homogénea, demostrando que el modelo comete imprecisiones suaves sin registrar predicciones catastróficas.
3. **Métricas Descartadas:** 
   * Se descartó **MAPE** debido a la presencia de días con ventas nulas (ventas reales = 0), lo que genera indeterminaciones por división entre cero, además de introducir un sesgo asimétrico cuando la demanda real es cercana a cero.
   * Se descartó **R-cuadrado (R²)** por medir únicamente porcentaje de varianza explicada, una métrica abstracta que no le indica al área de compras cuántos pallets pedir para abastecer las tiendas.

## 6. Arquitectura MLOps y Modularización
Para demostrar buenas prácticas de ingeniería de software para ML, el proyecto se estructuró separando claramente la fase de experimentación del entorno de producción. El código exploratorio fue modularizado en un pipeline funcional:

* **`notebooks/01_eda.ipynb`:** Contiene el Análisis Exploratorio de Datos (EDA) en crudo, la validación visual de hipótesis (como la detección del 100% de fallas en el POS) y el prototipado inicial.
* **`src/data_prep.py`:** Módulo encargado de la ingesta, limpieza, imputación e ingeniería de características, aislado para prevenir el filtrado de datos.
* **`src/train.py`:** Módulo que ejecuta el particionado temporal, el modelado y la evaluación.
* **`predict.py`:** Script que simula el pipeline de inferencia cargando el modelo campeón.
* **`main.py`:** Orquestador maestro para la ejecución secuencial del pipeline completo.
* **MLflow:** Implementado como *Tracking Server* para registrar hiperparámetros, métricas (RMSE, MAE) y serializar (congelar) el modelo automáticamente para su consumo.

## 7. Alcance del MVP y Roadmap para Reducción de Error (Next Steps)
El alcance de este entregable corresponde a un **Producto Mínimo Viable (MVP)** desarrollado en una ventana de 48 horas. Se priorizó la modularidad, la prevención de *Data Leakage* y un pipeline reproducible con un error promedio baseline de **MAE $\approx 204$ unidades**.

Para llevar este modelo a un entorno productivo de alto rendimiento y reducir el margen de error, se plantean las siguientes palancas analíticas en el Roadmap de desarrollo:

### A. Ingeniería de Características Avanzada (Feature Engineering)
1. **Ventanas Móviles y Rezagos Ampliados:** Incorporar *lags* de 14, 28 y 30 días, así como desviaciones estándar móviles (*Rolling Std*) para capturar la volatilidad intrínseca por tienda/categoría.
2. **Variables Macroeconómicas y Calendario Retail:** Integrar fechas de pago quincenales (días 15 y 30 en México), días festivos locales, temporalidades escolares (*Back to School*) y elasticidad de precios cuando los datos de promociones estén disponibles.
3. **Interacciones Jerárquicas:** Creación de *Target Encoding* y embeddings categóricos para capturar patrones a nivel Región-Tienda y Categoría-Producto.

### B. Optimización de Modelos y Ensembles
1. **Ajuste Fino de Hiperparámetros (Hyperparameter Tuning):** Implementar búsqueda bayesiana automatizada mediante **Optuna** dentro de MLflow, optimizando la tasa de aprendizaje (`learning_rate`), profundidad del árbol (`max_depth`) y regularización (`L1/L2`) para mitigar el sobreajuste.
2. **Ensemble & Stacking:** Combinar las predicciones de LightGBM con algoritmos complementarios (como CatBoost o XGBoost) utilizando un metamodelo lineal para promediar errores residuales.

### C. Operación y MLOps Productivo
1. **Estrategia Autorregresiva Multi-paso:** Implementar un esquema de predicción recursivo para proyectar horizontes de demanda a 7 y 14 días sin requerir observaciones intermedias reales.
2. **Monitoreo de Drift y Re-entrenamiento Continuo (CT):** Desplegar alertas automatizadas de *Data Drift* y *Concept Drift* vinculadas a un pipeline de re-entrenamiento periódico ejecutado mediante orquestadores (Airflow / Prefect).

## 8. Transparencia y Uso de Inteligencia Artificial (IA)
En cumplimiento con las buenas prácticas modernas de desarrollo, se declara el uso de un Modelo de Lenguaje (LLM) como asistente y *sparring* técnico durante este proyecto. Su integración aportó valor específico en las siguientes áreas:
* **Refactorización MLOps:** Asistencia en la estructuración del código exploratorio (Jupyter Notebook) hacia scripts modulares de producción orientados a objetos.
* **Brainstorming Arquitectónico:** Validación de estrategias de mitigación de *Data Leakage* y diseño conceptual del *Continuous Training* para mitigar el *Model Drift*.
* **Documentación:** Generación de la estructura del presente documento y el README, asegurando claridad comunicativa en el stack tecnológico.