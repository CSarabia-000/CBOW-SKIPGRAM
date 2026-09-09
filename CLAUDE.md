# Proyecto: Modelo CBOW sobre Spanish Billion Word Corpus (SBWC)

## Objetivo

Construir un modelo CBOW (Continuous Bag of Words) entrenado sobre el corpus limpio SBWC, comenzando con la configuración base **vocab=5,000 / dim=50 / contexto=2**, con arquitectura pensada para escalar fácilmente a otras combinaciones de la grilla:

- Vocabulario: 5k, 10k, 15k, 20k
- Dimensión de embeddings: 50, 100, 200, 300
- Tamaño de contexto: 2, 5, 10

**Entregables finales**: modelo(s) entrenado(s), embeddings exportados en formato Word2Vec estándar, vocabulario (mapeo palabra↔índice), y notebooks de validación/visualización.

---

## 1. Estructura de carpetas propuesta

```
cbow-sbwc/
├── data/
│   ├── raw/                  # sbwce.clean.txt.bz2 (no se descomprime a disco)
│   └── processed/            # vocabularios, pares generados (cache), por config
├── configs/
│   ├── base.yaml             # config 5k/50/2
│   └── grid.yaml             # define las variaciones a probar
├── src/
│   ├── __init__.py
│   ├── corpus.py             # lectura streaming del .bz2, tokenización
│   ├── vocabulary.py         # construcción de vocabulario + subsampling
│   ├── dataset.py            # generación de pares (contexto, target) + Dataset/DataLoader
│   ├── model.py               # arquitectura CBOW (PyTorch)
│   ├── train.py               # loop de entrenamiento, checkpoints
│   ├── evaluate.py            # vecinos cercanos, analogías, similitud coseno
│   └── export.py              # exportar embeddings a formato Word2Vec (.txt/.bin)
├── notebooks/
│   ├── 01_exploracion_corpus.ipynb
│   ├── 02_vocabulario.ipynb
│   ├── 03_generacion_pares.ipynb
│   ├── 04_entrenamiento.ipynb
│   ├── 05_evaluacion_embeddings.ipynb
│   └── 06_comparacion_configuraciones.ipynb
├── checkpoints/               # modelos guardados, uno por configuración
├── embeddings/                # embeddings exportados, uno por configuración
├── requirements.txt
└── README.md
```

**Principio de diseño clave**: cada notebook prueba UNA fase del pipeline de forma aislada, llamando funciones de `src/`, no reimplementando lógica dentro del notebook. Así los notebooks sirven como "pruebas interactivas" y el código real vive en módulos reutilizables y testeables.

---

## 2. Sistema de configuración (para hacerlo escalable desde el día 1)

Usar un archivo YAML simple por configuración, en vez de hardcodear hiperparámetros en el código:

```yaml
# configs/base.yaml
name: "base_5k_50_2"
vocab_size: 5000
embedding_dim: 50
context_size: 2
min_count: 5              # frecuencia mínima para entrar al vocabulario
subsampling_threshold: 1e-5
batch_size: 512
learning_rate: 0.003
epochs: 5
negative_samples: 10      # negative sampling en vez de softmax completo
corpus_sample_fraction: 1.0  # útil para pruebas rápidas con subset del corpus
seed: 42
```

Cada script (`train.py`, `evaluate.py`, etc.) recibe la ruta al YAML como parámetro. Para escalar, solo se crea un nuevo YAML (`configs/dim100_ctx5.yaml`, etc.) y se reutiliza todo el pipeline sin tocar código.

---

## 3. Librerías a instalar (ambiente virtual)

```bash
python -m venv venv
source venv/bin/activate   # en Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**`requirements.txt`:**

```
torch>=2.0.0
numpy>=1.24.0
pyyaml>=6.0
tqdm>=4.65.0
matplotlib>=3.7.0
scikit-learn>=1.3.0      # para PCA/t-SNE en visualización
jupyter>=1.0.0
ipykernel>=6.25.0
gensim>=4.3.0             # útil para comparar formato Word2Vec y validar exportación
spacy>=3.6.0               # tokenización en español (opcional, alternativa a regex simple)
```

Nota: si no se dispone de GPU, PyTorch en CPU es suficiente para la configuración base (5k/50/2); para configuraciones grandes (20k/300/10) se recomienda GPU si está disponible.

---

## 4. Plan de fases (para instruir a Claude Code)

### Fase 1 — `src/corpus.py`: Lectura del corpus
- Función `stream_sentences(path, sample_fraction=1.0)`: generador que abre el `.bz2` con `bz2.open(path, "rt", encoding="utf-8")` y produce oraciones (líneas) una por una, sin cargar todo en memoria.
- Función `tokenize(sentence)`: minúsculas + limpieza básica de puntuación + split en tokens.
- **Notebook 01**: leer las primeras N líneas, mostrar tokenización de ejemplo, contar total de líneas del corpus (usando streaming, no `wc -l` de shell si se quiere que quede reproducible en Python).

### Fase 2 — `src/vocabulary.py`: Construcción del vocabulario
- Función `build_vocabulary(corpus_path, vocab_size, min_count)`: cuenta frecuencias con streaming, se queda con las top-N palabras, agrega token `<UNK>`.
- Función `subsample(word_counts, threshold)`: implementa subsampling de palabras frecuentes (fórmula estándar de Mikolov et al.).
- Guarda el vocabulario en `data/processed/{config_name}/vocab.json` (palabra→índice e índice→palabra).
- **Notebook 02**: construir vocabulario con la config base, mostrar las 20 palabras más frecuentes, tamaño final del vocabulario, % de tokens cubiertos vs `<UNK>`.

### Fase 3 — `src/dataset.py`: Generación de pares contexto-target
- Función `generate_pairs(corpus_path, vocab, context_size, sample_fraction)`: generador que produce pares `(contexto, target)` como índices del vocabulario, deslizando la ventana sobre cada oración tokenizada.
- Clase `CBOWDataset(torch.utils.data.Dataset)` que envuelve los pares para usarlos con `DataLoader`.
- Importante: implementar como **generador/streaming** también aquí, no crear todos los pares en una lista en memoria si se va a usar el corpus completo.
- **Notebook 03**: generar pares con la config base sobre una muestra pequeña del corpus, imprimir 10 ejemplos de pares (contexto, target) ya traducidos de vuelta a palabras (no solo índices), para verificar visualmente que la ventana está bien construida.

### Fase 4 — `src/model.py`: Arquitectura CBOW
- Clase `CBOWModel(nn.Module)`:
  - `nn.Embedding(vocab_size, embedding_dim)` para el contexto.
  - Promedio de los embeddings del contexto.
  - Capa de salida con **negative sampling** (más eficiente que softmax completo, importante para cuando se escale a vocab=20k).
- **Notebook 04**: instanciar el modelo con la config base, hacer un forward pass de prueba con un batch dummy, verificar dimensiones de tensores en cada paso.

### Fase 5 — `src/train.py`: Entrenamiento
- Función `train(config_path)`: arma el pipeline completo (vocab → dataset → dataloader → modelo → loop de entrenamiento), guarda checkpoints en `checkpoints/{config_name}/`, registra la loss por época.
- Debe ser ejecutable tanto desde notebook como desde línea de comandos (`python -m src.train --config configs/base.yaml`).
- **Notebook 04** (continuación): entrenar la config base por pocas épocas, graficar la curva de loss.

### Fase 6 — `src/evaluate.py` y `src/export.py`: Validación y exportación
- Función `nearest_neighbors(word, embeddings, vocab, k=10)`: similitud coseno.
- Función `word_analogy(a, b, c, embeddings, vocab)`: para probar `rey - hombre + mujer ≈ reina`.
- Función `export_word2vec_format(embeddings, vocab, output_path)`: exporta a `.txt` con formato estándar (`vocab_size embedding_dim` en la primera línea, luego `palabra v1 v2 ... vn` por línea), compatible con `gensim.KeyedVectors.load_word2vec_format`.
- **Notebook 05**: cargar el modelo entrenado, probar vecinos cercanos de 5-10 palabras conocidas, probar 2-3 analogías, visualizar un subconjunto de embeddings con PCA/t-SNE en 2D.

### Fase 7 — Escalado y comparación
- **Notebook 06**: cargar métricas/resultados de varias configuraciones (una vez entrenadas) y compararlas lado a lado — ej. calidad de vecinos cercanos, tiempo de entrenamiento, tamaño en disco de los embeddings — para decidir cómo seguir escalando según el cómputo disponible.

---

## 5. Orden de escalado sugerido (aislando una variable a la vez)

1. `base_5k_50_2` (punto de partida)
2. `5k_100_2` → efecto de aumentar dimensión
3. `5k_100_5` → efecto de aumentar contexto
4. `10k_100_5` → efecto de aumentar vocabulario
5. `20k_200-300_10` → configuración final, solo si el cómputo lo permite

Cada corrida es solo un nuevo archivo YAML en `configs/`, sin tocar el código de `src/`.

---

## 6. Prompt sugerido para pasar a Claude Code

> Quiero que construyas el proyecto `cbow-sbwc` siguiendo esta especificación exacta (estructura de carpetas, módulos en `src/`, notebooks en `notebooks/`, sistema de configuración YAML). El corpus de entrada es `sbwce.clean.txt.bz2` (una oración por línea, sin duplicados), ubicado en `data/raw/`. Empieza implementando la Fase 1 (`src/corpus.py`) junto con `notebooks/01_exploracion_corpus.ipynb` para probarla, y avanza fase por fase, validando cada una con su notebook correspondiente antes de pasar a la siguiente. Usa PyTorch para el modelo, negative sampling en la capa de salida, y asegúrate de que todo el pipeline funcione por streaming (sin descomprimir el corpus completo a disco ni cargar todos los pares en memoria).
