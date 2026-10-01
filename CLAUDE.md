# Proyecto: word2vec (CBOW y skip-gram) sobre Spanish Billion Word Corpus (SBWC)

## Objetivo

Construir modelos word2vec entrenados sobre el corpus limpio SBWC. El proyecto arrancó con CBOW (Continuous Bag of Words) y **la Fase 8 agregó skip-gram** dentro del mismo pipeline; la arquitectura se elige con la llave `arch` de cada YAML. Se comenzó con la configuración base **vocab=5,000 / dim=50 / contexto=2**, con arquitectura pensada para escalar fácilmente a otras combinaciones de la grilla:

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
│   ├── model.py               # arquitecturas CBOW y skip-gram (PyTorch)
│   ├── train.py               # loop de entrenamiento, checkpoints
│   ├── evaluate.py            # vecinos cercanos, analogías, similitud coseno
│   └── export.py              # exportar embeddings a formato Word2Vec (.txt/.bin)
├── notebooks_CBOW/
│   ├── 00_resumen_proyecto.ipynb
│   ├── 01_exploracion_corpus.ipynb
│   ├── 02_vocabulario.ipynb
│   ├── 03_generacion_pares.ipynb
│   ├── 04_entrenamiento.ipynb
│   ├── 05_evaluacion_embeddings.ipynb
│   └── 06_comparacion_configuraciones.ipynb
├── notebooks_SKIPGRAM/        # mismo pipeline src/, otra arquitectura (arch: skipgram)
│   ├── 01_por_que_no_un_proyecto_nuevo.ipynb
│   ├── 02_skipgram.ipynb
│   ├── 03_skipgram_contexto5.ipynb
│   └── 04_resumen_comparativo.ipynb
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
---

## 7. Fase 8 — Skip-gram dentro del mismo proyecto

Skip-gram **no** es un proyecto aparte: vive en este repo, comparte corpus,
vocabulario, entrenamiento, evaluación y exportación con CBOW, y se elige con
una llave del YAML.

```yaml
# configs/piloto_sg_5k_50_2.yaml
arch: "skipgram"             # "cbow" (por defecto) o "skipgram"
vocab_from: "base_5k_50_2"   # el MISMO vocabulario que el piloto CBOW
```

### Por qué en el mismo proyecto y no en una carpeta nueva

Copiar el proyecto habría duplicado 7 de los 9 módulos de `src/` (corpus,
vocabulario, config, train, evaluate, export, compare son idénticos para las dos
arquitecturas), y con ellos sus futuros bugs. Pero la razón de fondo es la
comparación: para que CBOW y skip-gram sean comparables tienen que entrenar
sobre **el mismo vocabulario**, es decir el mismo espacio de índices. Eso ya lo
resolvía `vocab_from`, que solo funciona dentro de un mismo proyecto. Con dos
carpetas habría que copiar `vocab.json` a mano en cada refresco y confiar en que
nadie se olvide.

### Dónde está la diferencia entre las dos arquitecturas

En **los pares**, no en el modelo. Sobre la misma ventana:

* CBOW: un par por token — `([a, b, d, e], c)`.
* Skip-gram: un par por vecino — `([c], a)`, `([c], b)`, `([c], d)`, `([c], e)`.

`generate_skipgram_pairs` se apoya en `generate_pairs` y expande cada par, en vez
de recorrer el corpus por su cuenta: así el muestreo de líneas, el subsampling,
`drop_unknown`, el corte de validación y el reparto entre workers son
literalmente el mismo código, y las dos arquitecturas no pueden terminar
entrenando sobre ventanas distintas sin que nadie se dé cuenta.

En `model.py` la diferencia cabe en un `forward`: `Word2VecModel` tiene las dos
matrices, el negative sampling y la loss; `CBOWModel` promedia el contexto con
`masked_mean` y `SkipGramModel` devuelve el embedding del centro. El contrato del
batch es `(entrada, máscara, target)` en las dos, con la entrada de ancho
`2 * context_size` en CBOW y 1 en skip-gram, así que `train.py` no ramifica nunca.

### Tres cosas que hay que tener presentes

1. **La ventana es fija**, no sorteada por token como en el word2vec original.
   CBOW ya se había entrenado así; mantenerlo igual en las dos es lo que hace que
   la comparación aísle la arquitectura.
2. **Una época de skip-gram cuesta ~3.5x más** con `context_size=2` (medido: 300
   oraciones dan 737 pares en CBOW y 2.042 en skip-gram). Es costo esperado, no
   un problema de configuración.
3. **La loss NO se compara entre arquitecturas.** No son versiones más fácil o
   más difícil de la misma tarea, son tareas distintas sobre conjuntos de pares
   de tamaños distintos. `comparable_loss` en `compare.py` ya incluye `arch` para
   que las dos familias no se crucen por accidente, y `cross_validation_loss`
   falla si se le pasan configuraciones de distinta arquitectura. **La
   comparación válida es la precisión en analogías**, y mejor todavía `mcnemar`
   sobre los mismos ítems.

### Configuraciones

`piloto_sg_5k_50_2` es el gemelo exacto de `piloto_5k_50_2`: cambia `arch` y
nada más. Cualquier configuración anterior a esta fase no declara `arch` y
`load_config` le pone `"cbow"`, así que las corridas ya entrenadas siguen
funcionando sin tocar sus YAML.

### Resultado medido (no repetir el experimento esperando otra cosa)

**Empate**: 40,2% skip-gram contra 41,3% CBOW en analogías, McNemar p = 0,74, con
2,77x los pares y 2,09x el tiempo. Ninguna categoría se salva.

La hipótesis de que skip-gram gana con palabras poco frecuentes **no se pudo
probar acá**, y la razón importa para cualquier repetición: con vocabulario de
5.000 sobre 2.028M de tokens no hay palabras raras —la última del vocabulario
(`cercana`) aparece 33.838 veces—, así que el mecanismo no tiene dónde actuar. La
prueba real pide vocabulario 20k+ sobre el corpus completo. No tiene sentido
volver a correr esta misma comparación con otra semilla o más épocas esperando que
cambie.

También ojo con desagregar por frecuencia: los rangos mezclan categorías de
analogía en proporciones muy distintas (el rango 1-500 es 54% `plural`, el
3000-5000 es 100% `país → gentilicio`), así que los niveles entre rangos no son
comparables. Solo vale la comparación pareada dentro de cada rango.

### Costo por época: usar la mediana, no el promedio

`history.json` guarda reloj de pared. Si la máquina se suspende a mitad de una
época, esa época queda con las horas de siesta adentro. Pasó en las dos corridas
comparadas (una época de 1.151s en `piloto_5k_50_2`, una de 7.294s en
`piloto_sg_5k_50_2`) y en el primer caso llegó a invertir la conclusión de costo
de la Fase 7, haciendo parecer que la corrida más chica era la más lenta.
`run_summary` devuelve `seconds_per_epoch` (promedio, como siempre) y
`seconds_per_epoch_median`; para costo, usar la mediana.


---

## 8. Fase 9 — Arquitectura × contexto, y el resumen comparativo

`configs/piloto_sg_5k_50_5.yaml` completa un **2×2 de arquitectura × contexto**
(CBOW/skip-gram × ctx 2/5, todo lo demás fijo). La pregunta que solo ese cuadro
podía contestar: la Fase 7 midió que a CBOW la ventana ancha no le sirve; ¿le
sirve a skip-gram, que no promedia el contexto sino que trata cada vecino como un
ejemplo propio?

### Resultado medido (no repetir esperando otra cosa)

**La predicción falló en la dirección opuesta.** Ensanchar de ctx 2 a 5 da +1,0 pp
en CBOW (p = 0,66) y **−3,5 pp en skip-gram** (p = 0,16). Y apareció el **único
efecto de arquitectura significativo del proyecto, en contra de skip-gram**: con
contexto 5 queda 5,6 pp por debajo de CBOW (IC [−10,2, −1,0], p = 0,026), con
`plural` cayendo de 31,1% a 21,1% (p = 0,035).

El razonamiento previo estaba invertido, y conviene dejarlo anotado: el argumento
suponía que el vecino lejano trae señal. Si trae ruido —y la morfología del
español vive en el vecino inmediato—, **promediar es una defensa**. CBOW diluye
el ruido; skip-gram le da peso completo a cada vecino lejano, y por eso se
degrada más.

Cautela obligatoria: con Bonferroni sobre las ocho comparaciones del proyecto el
umbral es 0,00625. Los dos efectos de dimensión lo pasan; este **no**. Es señal
consistente, no conclusión cerrada.

### Costo: skip-gram escala con la ventana, CBOW no

CBOW genera un par por token sin importar el ancho (5.009.035/época a ctx 2 y a
ctx 5). Skip-gram genera uno por vecino: 13.865.356 a ctx 2 (**2,77×**) y
24.968.216 a ctx 5 (**4,98×**), o sea 7,0 min/época contra 2,7 de CBOW. Ensanchar
la ventana en skip-gram es lo más caro que hace el proyecto y da el peor
resultado.

### `paired_difference` en `compare.py`

`mcnemar` contesta **si** dos modelos difieren; `paired_difference` contesta
**cuánto**, con un IC calculado sobre los pares. Los ítems que ambos modelos
aciertan o fallan no aportan a la diferencia, y tratarlos como dos muestras
independientes infla el error estándar hasta borrar efectos reales. Reportar un
efecto pide las dos cosas.

### Lo que quedó establecido, de ocho decisiones medidas

Solo **una rindió**: duplicar la dimensión (+12,9 pp, p < 0,0001). Dos salieron
en contra (ensanchar con dim 100; skip-gram con ctx 5) y cinco son
indistinguibles de cero. De las seis corridas probadas contra la línea de base,
**cinco tienen su intervalo superpuesto con el de ella**; la única que se despega
es `piloto_5k_100_2`, que además es la más barata y la mejor a la vez.

`notebooks_SKIPGRAM/04_resumen_comparativo.ipynb` tiene la tabla de las siete corridas, el
forest plot de los ocho efectos y las tres hipótesis del proyecto que las
mediciones refutaron.

### Lo que sigue sin probarse

Subir el contexto ya se probó tres veces (CBOW dim 50, CBOW dim 100, skip-gram) y
ninguna ganó. **Bajarlo a `context_size: 1` nunca se probó**, y es ahora la
predicción más directa que queda abierta: si la ventana angosta favorece la
forma, ahí debería verse.
