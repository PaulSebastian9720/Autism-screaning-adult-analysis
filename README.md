# Proyecto: Práctica 4 — Redes Neuronales y Transformaciones

Resumen breve
- Proyecto de prácticas con notebooks y scripts para entrenar y evaluar modelos de redes neuronales (Keras/TensorFlow). Incluye datos (`.arff`, `.csv`), notebooks Jupyter, y artefactos de MLflow.

Archivos relevantes
- `IA_FNT_CLB_OPM_RNN.ipynb` — Notebook principal de entrenamiento/experimentos.
- `2_IA_Interciclo_Transformaciones.ipynb` — Notebook de transformaciones y preprocesado.
- `dataset_transformado_con_etiquetas.csv` / `dataset_transformado_sin_etiquetas.csv` — Datos procesados.
- `mejor_modelo.h5` — Modelo Keras guardado (pesos + arquitectura).
- `requirements.txt` — Lista de dependencias pip (usar para replicar el entorno).

Versión de Python recomendada
- Basado en el entorno y dependencias presentes en `venv`, se recomienda usar Python 3.10.x (por ejemplo `3.10.14` o `3.10.13`).

Pasos para replicar el entorno (opción A: pyenv + virtualenv)
1. Instalar `pyenv` . En Linux, seguir la guía oficial: https://github.com/pyenv/pyenv
2. Instalar la versión recomendada y crear entorno virtual:

```bash
pyenv install 3.10.14
pyenv virtualenv 3.10.13 practicas-ia-3.10
pyenv local practicas-ia-3.10
python -V
# debe mostrar Python 3.10.14
python -m venv venv
source venv/bin/activate
```

3. Instalar dependencias:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Registrar kernel para Jupyter:

```bash
pip install ipykernel
python -m ipykernel install --user --name=practicas-ia-3.10 --display-name "practicas-ia-3.10"
```

Pasos para replicar el entorno (opción B: `python -m venv` sin pyenv)
1. Crear y activar venv con la versión del sistema (si ya tienes Python 3.10):

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Ejecutar los notebooks
- Abrir Jupyter Lab / Notebook:

```bash
pip install jupyterlab
jupyter lab
```

- Abrir `IA_FNT_CLB_OPM_RNN.ipynb` y ejecutar celdas. Si usas MLflow local, los runs quedarán en `./mlruns`.

Uso del modelo guardado (`mejor_modelo.h5`)
- Cargar el modelo para inferencia:

```python
from tensorflow.keras.models import load_model
model = load_model('mejor_modelo.h5')
# preparar tus datos y usar model.predict(X)
```

MLflow
- Los experimentos se registran en `mlruns/` (tracking local por defecto). Para cambiar el tracking URI en los notebooks, busca líneas con `mlflow.set_tracking_uri(...)`.

Consejos y buenas prácticas
- Evita commitear la carpeta `venv/` ni `mlruns/` al repositorio (ya está en `.gitignore`).
- Mantén `requirements.txt` actualizado: después de instalar paquetes nuevos, ejecutar `pip freeze > requirements.txt` desde el entorno activo.
- Para reproducibilidad avanzada, considera exportar un `conda`/`mamba` env con versiones exactas.


