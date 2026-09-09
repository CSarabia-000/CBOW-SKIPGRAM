# word2vec-sbwc — CBOW y skip-gram sobre el Spanish Billion Word Corpus

Entrenamiento de embeddings de palabras con **word2vec + negative sampling**
(PyTorch) sobre el corpus limpio SBWC (`sbwce.clean.txt.bz2`, 2,97 GB comprimido,
una oración por línea), desde la lectura del `.bz2` hasta la exportación en
formato Word2Vec compatible con gensim.

Las **dos arquitecturas** comparten el pipeline completo y se eligen con una
llave del YAML: `arch: "cbow"` (por defecto) o `arch: "skipgram"`. El proyecto
nació como CBOW; la Fase 8 agregó skip-gram sin duplicar nada.

Configuración de partida: **vocab=5.000 / dim=50 / contexto=2**. Todo el pipeline
funciona **por streaming** (nunca se descomprime el corpus a disco ni se
materializa la lista de pares) y está preparado para escalar a la grilla
`vocab ∈ {5k, 10k, 15k, 20k}`, `dim ∈ {50, 100, 200, 300}`,
`contexto ∈ {2, 5, 10}` **sin tocar el código de `src/`**: solo se agrega un YAML
nuevo en `configs/`.

---

## Estado del proyecto

| Fase | Módulo | Notebook | Estado |
|------|--------|----------|--------|
| 1. Lectura del corpus | `src/corpus.py` | `01_exploracion_corpus.ipynb` | ✅ implementada |
| 2. Vocabulario | `src/vocabulary.py` | `02_vocabulario.ipynb` | ✅ implementada |
| 3. Pares contexto-target | `src/dataset.py` | `03_generacion_pares.ipynb` | ✅ implementada |
| 4. Modelo (CBOW y skip-gram) | `src/model.py` | `04_entrenamiento.ipynb` | ✅ implementada |
| 5. Entrenamiento | `src/train.py` | `04_entrenamiento.ipynb` | ✅ implementada |
| 6. Evaluación y exportación | `src/evaluate.py`, `src/export.py` | `05_evaluacion_embeddings.ipynb` | ✅ implementada |
| 7. Comparación de configuraciones | `src/compare.py` | `06_comparacion_configuraciones.ipynb` | ✅ implementada |
| 8. Skip-gram | `src/dataset.py`, `src/model.py` | *(pendiente: 07)* | ⚙️ implementada, sin entrenar |

**Artefactos ya producidos**:

- `data/processed/base_5k_50_2/vocab.json` — vocabulario de 5.000 palabras
  construido sobre el **corpus completo**
- `data/processed/muestra_2m.txt` — muestra aleatoria de 2M de oraciones (242 MB)
- **cinco corridas entrenadas** (4 de la grilla + 1 ablación), cada una con sus
  15 checkpoints, `best.pt`, `last.pt`, `history.json` y sus embeddings
  exportados en `.txt` y `.bin`, validados con gensim

> **Configuración recomendada: `piloto_5k_100_2`** (dim 100, contexto 2), la
> mejor de la grilla con 54,2% en analogías. Ver *Fase 7*.

> **Por dónde empezar a leer:** `notebooks/00_resumen_proyecto.ipynb` recorre las
> primeras seis fases con un ejemplo y un gráfico por fase. Los notebooks 01-05
> tienen el detalle de cada una, y el 06 es la comparativa de la Fase 7.

> **Estado de la Fase 8 (skip-gram):** el código está listo y probado de punta a
> punta, pero **todavía no hay ninguna corrida skip-gram entrenada**. La
> configuración `configs/piloto_sg_5k_50_2.yaml` es el gemelo exacto del piloto
> CBOW (cambia `arch` y nada más) y está lista para lanzarse. Ver *Fase 8*.

---

## Estructura

```
proyecto_cbow/
├── configs/
│   ├── base.yaml                  # corrida sobre el corpus completo (95,8M oraciones)
│   ├── piloto_5k_50_2.yaml        # corrida piloto sobre la muestra de 2M
│   ├── piloto_5k_100_2.yaml       # grilla fase 7: dim 100
│   ├── piloto_5k_50_5.yaml        # grilla fase 7: contexto 5
│   ├── piloto_5k_100_5.yaml       # grilla fase 7: dim 100 + contexto 5
│   ├── ablacion_unk_5k_50_2.yaml  # ablación: entrenar CON <UNK>
│   └── piloto_sg_5k_50_2.yaml     # fase 8: skip-gram, gemelo de piloto_5k_50_2
├── data/
│   ├── raw/sbwce.clean.txt.bz2    # el corpus (no se versiona)
│   └── processed/
│       ├── base_5k_50_2/vocab.json
│       └── muestra_2m.txt
├── src/
│   ├── __init__.py                # configure_console(): fuerza UTF-8 en la consola
│   ├── config.py                  # carga de YAML + rutas derivadas de cada corrida
│   ├── corpus.py                  # Fase 1: streaming del .bz2 + tokenización
│   ├── vocabulary.py              # Fase 2: frecuencias, poda, <UNK>, subsampling
│   ├── dataset.py                 # Fase 3: pares CBOW y skip-gram, Dataset e IterableDataset
│   ├── model.py                   # Fase 4: Word2VecModel -> CBOWModel / SkipGramModel
│   ├── train.py                   # Fase 5: loop, validación, checkpoints, history
│   ├── evaluate.py                # Fase 6: vecinos, analogías, precisión
│   ├── export.py                  # Fase 6: exportación a formato Word2Vec
│   └── compare.py                 # Fase 7: tabla comparativa, Wilson, McNemar
├── notebooks/                     # 00_resumen + 01..06, uno por fase
├── checkpoints/{name}/            # modelos guardados, una carpeta por corrida
├── embeddings/{name}.{txt,bin}    # embeddings exportados
├── requirements.txt
└── README.md
```

Cada corrida se identifica por el campo `name` de su YAML, y ese nombre determina
todas sus rutas: `data/processed/{name}/`, `checkpoints/{name}/`,
`embeddings/{name}.txt`. Dos corridas nunca se pisan.

---

## Instalación

Requiere **Python 3.10+** (probado en 3.13). GPU opcional.

```bash
python -m venv venv
venv\Scripts\activate              # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

Si hay GPU NVIDIA, instalar PyTorch con CUDA (en esta máquina, una **RTX 3060
Laptop de 6 GB**):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Sin GPU el pipeline funciona igual en CPU; la configuración base (5k/50/2) es
perfectamente entrenable así, solo más lento.

Dependencias por fase: para las Fases 1-2 alcanza con `pyyaml`, `tqdm`,
`matplotlib` y `jupyter`; `torch` hace falta desde la Fase 3, y `gensim` /
`scikit-learn` desde la Fase 6.

### El corpus

Va en `data/raw/sbwce.clean.txt.bz2`. **No se versiona ni se descomprime**: todos
los módulos lo leen comprimido, línea a línea. `src/corpus.open_corpus` también
acepta `.gz` y texto plano, así que cualquiera de los tres formatos funciona.

---

## Inicio rápido

### A) Usar los embeddings ya entrenados

Si solo se quieren los vectores, ya están exportados y no hace falta ni el corpus
ni PyTorch:

```python
from gensim.models import KeyedVectors

kv = KeyedVectors.load_word2vec_format("embeddings/piloto_5k_50_2.txt")
kv.most_similar("febrero", topn=5)
# [('marzo', 0.98), ('abril', 0.98), ('octubre', 0.98), ('mayo', 0.97), ...]

kv.most_similar(positive=["mujer", "rey"], negative=["hombre"], topn=3)
```

O con el código del proyecto, que evita depender de gensim:

```python
from src.config import load_config
from src.evaluate import load_embeddings, nearest_neighbors, word_analogy

config = load_config("configs/piloto_5k_50_2.yaml")
embeddings, vocab = load_embeddings(config)            # usa checkpoints/{name}/best.pt

nearest_neighbors("rojo", embeddings, vocab, k=5)
word_analogy("hombre", "rey", "mujer", embeddings, vocab)
```

### B) Reproducir la corrida piloto de punta a punta

Cuatro pasos, ~1h20 en total en la RTX 3060 (los tiempos están medidos):

```bash
# 1) Vocabulario sobre el corpus COMPLETO           (~46 min, un pase por los 3 GB)
python -m src.vocabulary --config configs/base.yaml --max-types 3000000

# 2) Muestra aleatoria de 2M de oraciones            (~6 min, 242 MB)
python -m src.corpus --sample-to data/processed/muestra_2m.txt \
                     --sample-fraction 0.020877

# 3) Entrenamiento: 15 épocas, 75,2M de pares        (~57 min)
python -m src.train --config configs/piloto_5k_50_2.yaml

# 4) Evaluación y exportación                        (segundos)
python -m src.evaluate --config configs/piloto_5k_50_2.yaml
python -m src.export   --config configs/piloto_5k_50_2.yaml --verify
python -m src.export   --config configs/piloto_5k_50_2.yaml --binary --verify
```

El paso 1 es el caro y **se hace una sola vez**: el piloto no reconstruye
vocabulario, lo toma prestado (ver *Corrida piloto*, más abajo).

Para una prueba de humo en minutos, sin el corpus completo, se puede acotar todo
con `--limit` y `--max-steps-per-epoch`:

```bash
python -m src.vocabulary --config configs/base.yaml --limit 200000 \
                         --output data/processed/prueba/vocab.json
python -m src.train --config configs/piloto_5k_50_2.yaml --max-steps-per-epoch 50
```

### C) Entrenar una configuración nueva

```bash
cp configs/piloto_5k_100_2.yaml configs/piloto_5k_200_2.yaml
# editar: name: "piloto_5k_200_2"  y  embedding_dim: 200
python -m src.train    --config configs/piloto_5k_200_2.yaml
python -m src.evaluate --config configs/piloto_5k_200_2.yaml
python -m src.export   --config configs/piloto_5k_200_2.yaml --verify

# y compararla contra las que ya existen
python -m src.compare --configs piloto_5k_100_2 piloto_5k_200_2
```

Nada más. No se toca `src/`, y como la nueva corrida hereda `vocab_from`, entrena
en el mismo espacio de índices y sus métricas son directamente comparables contra
las cinco corridas que ya están.

---

## Referencia de línea de comandos

Todos los módulos son ejecutables (`python -m src.<módulo>`) y todos aceptan
`--config`, salvo `src.corpus`, que trabaja sobre un archivo suelto.

| Comando | Qué hace | Opciones |
|---|---|---|
| `python -m src.corpus` | Estadísticas del corpus: oraciones, tokens, longitudes, ejemplos de tokenización | `--path`, `--limit N`, `--sample-fraction F`, `--seed` |
| `python -m src.corpus --count-lines` | Cuenta las líneas del corpus completo (recorre los 3 GB) | `--path` |
| `python -m src.corpus --sample-to ARCHIVO` | Materializa una muestra aleatoria **barajada** a un `.txt` | `--sample-fraction F`, `--seed`, `--no-shuffle` |
| `python -m src.vocabulary` | Construye el vocabulario → `data/processed/{name}/vocab.json` | `--limit N`, `--max-types N`, `--output RUTA`, `--top N` |
| `python -m src.dataset` | Genera pares y muestra ejemplos traducidos a palabras | `--limit N`, `--show N` |
| `python -m src.model` | Instancia el modelo y hace un forward de prueba mostrando dimensiones | `--batch-size N` |
| `python -m src.train` | Entrena → `checkpoints/{name}/` | `--device cuda\|cpu`, `--num-workers N`, `--resume`, `--max-steps-per-epoch N` |
| `python -m src.evaluate` | Vecinos cercanos, analogías y precisión sobre el set de analogías | `--checkpoint`, `--k N`, `--words p1 p2 ...` |
| `python -m src.export` | Exporta a formato Word2Vec → `embeddings/{name}.txt` | `--binary`, `--verify`, `--normalize`, `--include-unk`, `--checkpoint` |
| `python -m src.compare` | Compara varias corridas: tabla, precisión por categoría, McNemar, vecinos lado a lado | `--configs a b c`, `--k N`, `--words ...`, `--json RUTA` |

Notas útiles:

- **`--max-types`** acota los tipos distintos que se mantienen en memoria durante
  el conteo, podando la cola larga (misma estrategia que gensim). Con un techo de
  ≥10× el `vocab_size` el resultado es bit a bit idéntico a un conteo exacto: no
  afecta a las palabras frecuentes, que son las únicas que entran al vocabulario.
- **`--resume`** retoma desde `last.pt`, restaurando también el estado del
  optimizador.
- **`--verify`** recarga el archivo exportado con gensim y muestra sus vecinos,
  como validación de que el formato quedó bien.
- **`--num-workers`** existe pero en esta máquina **empeora** el rendimiento
  (43k → 25k pares/s de 0 a 3 workers): el costo de arrancar procesos y serializar
  batches supera lo que se gana paralelizando la tokenización. `0` es la
  recomendación.

---

## API de Python

Los notebooks no reimplementan nada: llaman a estas funciones.

```python
# --- Fase 1: corpus ---
from src.corpus import stream_sentences, stream_tokens, tokenize, write_sample, corpus_stats

for tokens in stream_tokens("data/raw/sbwce.clean.txt.bz2", limit=1000):
    ...                                    # sin descomprimir el archivo entero
tokenize("El 12/03/1998 llegó a Buenos-Aires.")
# ['el', '<NUM>', 'llegó', 'a', 'buenos-aires']

# --- Fase 2: vocabulario ---
from src.config import load_config, vocab_path
from src.vocabulary import Vocabulary, build_vocabulary_from_config, subsample

config = load_config("configs/base.yaml")
vocab = Vocabulary.load(vocab_path(config))
vocab.encode(["el", "gobierno", "anuncio", "quijote"])   # -> [4, 135, 3974, 0]
                                                 #    0 = <UNK>: "quijote" no entró
vocab.coverage                                   # 0.8326  (propiedad, no método)
vocab.most_common(3)                             # [('de', 151730127), ('la', 81271721), ...]

# --- Fase 3: pares ---
from src.dataset import generate_pairs, pad_context, CBOWDataset, CBOWIterableDataset

dataset = CBOWIterableDataset.from_config(config, vocab)   # streaming
dataset = CBOWDataset.from_config(config, vocab, max_pairs=100_000)  # en memoria

# --- Fase 4-5: modelo y entrenamiento ---
from src.model import build_model_from_config
from src.train import train, load_checkpoint, load_history

model, sampler = build_model_from_config(config, vocab)
train("configs/piloto_5k_50_2.yaml")
historia = load_history(config)                  # curvas, sin cargar el modelo

# --- Fase 6: evaluación y exportación ---
from src.evaluate import load_embeddings, nearest_neighbors, word_analogy, evaluate_analogies
from src.export import export_from_config

export_from_config(config, binary=True)

# --- Fase 7: comparación ---
from src.compare import GRID, collect, cross_validation_loss, mcnemar, wilson_interval

filas = collect(GRID)                            # una fila por corrida
wilson_interval(118, 286)                        # (0.357, 0.470)
mcnemar(hits_a, hits_b)                          # prueba pareada sobre los mismos ítems
cross_validation_loss("configs/ablacion_unk_5k_50_2.yaml",
                      "configs/piloto_5k_50_2.yaml")   # misma vara para los dos
```

---

## Fase 7 — Comparación de configuraciones

Factorial 2×2 (dim ∈ {50, 100} × contexto ∈ {2, 5}) con **todo lo demás fijo**:
mismo vocabulario prestado, mismas 2M de oraciones, mismas 15 épocas, mismo lr,
misma semilla. Las cuatro corridas tardaron 2h47 en total.

| Corrida | dim | ctx | val loss | analogías | IC 95% | min/época | checkpoint |
|---|---|---|---|---|---|---|---|
| `piloto_5k_50_2` | 50 | 2 | 2,6430 | 41,3% | [35,7%, 47,0%] | 3,8 | 5,7 MB |
| **`piloto_5k_100_2`** | **100** | **2** | **2,6128** | **54,2%** | [48,4%, 59,9%] | 2,6 | 11,4 MB |
| `piloto_5k_50_5` | 50 | 5 | 2,6183 | 42,3% | [36,7%, 48,1%] | 2,7 | 5,7 MB |
| `piloto_5k_100_5` | 100 | 5 | 2,5905 | 49,7% | [43,9%, 55,4%] | 2,8 | 11,4 MB |

Las diferencias se contrastan con **McNemar pareado** sobre los mismos 286 ítems,
no comparando dos porcentajes: los cuatro modelos responden exactamente las mismas
analogías, así que lo que importa es en cuáles discrepan.

### La dimensión es el factor que manda

| efecto | cambio | McNemar |
|---|---|---|
| dim 50→100, con ctx 2 | **+12,9 pp** | p < 0,0001 |
| dim 50→100, con ctx 5 | **+7,3 pp** | p = 0,0003 |
| ctx 2→5, con dim 50 | +1,0 pp | p = 0,66 (nada) |
| ctx 2→5, con dim 100 | **−4,5 pp** | p = 0,035 (**empeora**) |

Y mejora justo donde el modelo era débil — las categorías morfológicas:

| categoría | dim 50 | dim 100 | p |
|---|---|---|---|
| verbo presente→pasado | 19,6% | **39,3%** | 0,0034 |
| género | 61,1% | **74,4%** | 0,0042 |
| plural | 27,8% | **38,9%** | 0,0213 |
| país → gentilicio | 76,7% | **96,7%** | 0,0312 |
| adjetivo → adverbio | 20,0% | 10,0% | 0,50 (n=20, ruido) |

Duplicar la dimensión duplica los parámetros y el archivo exportado, pero **no
cuesta tiempo**. Las tres corridas nuevas se ejecutaron encadenadas en idénticas
condiciones y las tres rondan los 2,6-2,8 min/época, sin importar la dimensión ni
el contexto; el cuello de botella es la CPU generando pares —el mismo trabajo en
las cuatro—, no la GPU. (Los 3,8 min de la línea de base se midieron en otra
sesión y no son comparables con esos tres.)

### Una hipótesis refutada

Al cerrar la Fase 6 dejé escrito que la debilidad morfológica venía de
`context_size=2` y que subir el contexto la arreglaría. **El experimento dice que
no**, en los dos sentidos: subir el contexto empeoró la morfología (con dim 100
bajaron las tres categorías morfológicas), y la causa real era la **dimensión**.
El razonamiento de fondo también estaba invertido: una ventana angosta captura
*más* sintaxis, no menos — el vecino inmediato es el que lleva la información de
forma. Una ventana ancha captura relaciones temáticas, y de hecho lo único que
mejoró al ensanchar fue `país → gentilicio` (76,7% → 90,0% con dim 50), mientras
hubo margen: con dim 100 ya estaba en 96,7% y no se movió.

### Menor loss ≠ mejor modelo

`piloto_5k_100_5` tiene la **loss más baja** de las cuatro (2,5905) y **no** es la
mejor en analogías (49,7% contra 54,2%). Eligiendo por la curva de entrenamiento
habríamos elegido mal. Además, la loss solo es comparable entre corridas del mismo
`context_size`: predecir desde 10 palabras es más fácil que desde 4, así que baja
por la tarea, no por el modelo.

## Ablación: entrenar con `<UNK>`

`ablacion_unk_5k_50_2` es idéntica a `piloto_5k_50_2` salvo `drop_unknown: false`,
que convierte los OOV en un token comodín en vez de eliminarlos.

**Primero, la magnitud real del cambio.** `<UNK>` es el 16,75% del corpus crudo,
pero el subsampling lo castiga justamente por frecuente: sobrevive solo el
**0,70%** de sus apariciones (la fórmula `sqrt(t/f)` predecía 0,77%). Termina
siendo el **0,97%** del stream de entrenamiento — aunque, curiosamente, sigue
siendo el token que el modelo más ve, 1,45× más que `de`.

| | sin `<UNK>` | con `<UNK>` | ¿diferencia real? |
|---|---|---|---|
| loss sobre la **misma** validación | 2,6430 | 2,6418 | no (0,0012) |
| analogías | 41,3% | 44,4% | **no** (McNemar p = 0,18) |
| pares por época | 5.009.035 | 5.067.178 | +1,16% |

> La loss de validación *propia* de la corrida con `<UNK>` es más baja (2,6170 vs
> 2,6430), pero eso es un artefacto: su validación incluye `<UNK>`, y predecir el
> token más frecuente es fácil. Puestos los dos modelos sobre la misma validación
> sin `<UNK>`, la diferencia desaparece.

**El experimento no respalda que sea mejor entrenar sin `<UNK>`.** Las dos
configuraciones son estadísticamente indistinguibles, y el único movimiento
apreciable va *a favor* de `<UNK>`. La razón es la que muestra la medición: el
subsampling ya elimina el 99,3% de los `<UNK>`, así que `drop_unknown` decide el
destino del 0,7% que queda — demasiado poco para mover la aguja.

Se mantiene `drop_unknown: true`, pero por razones de **costo e higiene**, no de
calidad:

1. Cuesta 1,16% más de cómputo por época a cambio de una mejora indistinguible de cero.
2. Vuelve la loss engañosa y obliga a montar una evaluación cruzada para comparar
   con cualquier otra corrida.
3. `<UNK>` no es una palabra: su vector es el promedio de todo lo que quedó afuera,
   y hay que acordarse de excluirlo en cada consulta y en la exportación.

Y hay un argumento que **queda retirado**: en la Fase 6 escribí que dejar `<UNK>`
haría que el modelo "gastara capacidad prediciendo un token comodín". A 0,97% del
stream esa capacidad es despreciable, y la medición lo confirma. No era el motivo.

## Fase 8 — Skip-gram

Skip-gram vive en este mismo proyecto, no en una carpeta aparte: comparte corpus,
vocabulario, entrenamiento, evaluación, exportación y comparación con CBOW.

```bash
python -m src.train --config configs/piloto_sg_5k_50_2.yaml
```

### La diferencia está en los pares, no en el modelo

Sobre la **misma ventana** (`context_size=2`), las dos arquitecturas la leen en
direcciones opuestas:

```
oración:   ... sudán palestinos regresar hogares la ...

CBOW        [sudán palestinos hogares la]  ->  regresar     (1 par por token)
skip-gram   regresar  ->  [sudán]
            regresar  ->  [palestinos]                       (1 par por vecino)
            regresar  ->  [hogares]
            regresar  ->  [la]
```

Medido sobre 300 oraciones: **737 pares en CBOW, 2.042 en skip-gram** (2,8x). Por
eso una época de skip-gram cuesta varias veces más sobre el mismo texto. A cambio,
cada par pone a la palabra central sola del lado de la entrada, en vez de diluida
en el promedio de su contexto — que es la razón por la que skip-gram suele rendir
mejor con palabras poco frecuentes.

`generate_skipgram_pairs` se apoya en `generate_pairs` y expande cada par, en vez
de recorrer el corpus por su cuenta: el muestreo de líneas, el subsampling,
`drop_unknown`, el corte de validación y el reparto entre workers son el mismo
código en las dos arquitecturas. En `model.py`, `Word2VecModel` tiene las dos
matrices, el negative sampling y la loss; lo único que redefinen `CBOWModel` y
`SkipGramModel` es el `forward` (promedio enmascarado vs. embedding del centro).

### Cómo comparar CBOW con skip-gram (y cómo no)

**La loss no se compara entre arquitecturas.** No son versiones más fácil o más
difícil de la misma tarea: son tareas distintas, sobre conjuntos de pares de
tamaños distintos. `comparable_loss` incluye `arch` para que las dos familias no
se crucen por accidente, y `cross_validation_loss` falla si se le pasan
configuraciones de distinta arquitectura.

**La comparación válida es la precisión en analogías**, y mejor todavía `mcnemar`
sobre los mismos ítems, que es mucho más sensible que comparar dos porcentajes con
sus intervalos:

```python
from src.compare import collect, mcnemar

filas = collect(["piloto_5k_50_2", "piloto_sg_5k_50_2"], detail=True)
print(mcnemar(filas[0]["items"], filas[1]["items"]))
```

Para que esa comparación signifique algo, las dos corridas comparten
`vocab_from: base_5k_50_2` — el mismo vocabulario, o sea el mismo espacio de
índices— y son idénticas en todo lo demás.

---

## Configuración

Cada corrida se define por completo en un YAML. Los hiperparámetros no se
hardcodean en ningún lado.

```yaml
name: "piloto_5k_50_2"          # define TODAS las rutas de la corrida
vocab_from: "base_5k_50_2"      # (opcional) reutiliza el vocab.json de otra corrida

corpus_path: "data/processed/muestra_2m.txt"
corpus_sample_fraction: 1.0     # <1.0 = muestreo aleatorio de líneas
tokenizer:
  lowercase: true
  number_token: "<NUM>"         # null para descartar números en vez de normalizarlos
  min_length: 1

vocab_size: 5000                # TOTAL, incluyendo <UNK>
min_count: 5                    # frecuencia mínima para entrar al vocabulario
subsampling_threshold: 1.0e-5   # descarte probabilístico de palabras muy frecuentes

context_size: 2                 # palabras a cada lado del target
drop_unknown: true              # sacar los OOV de la oración en vez de dejarlos como <UNK>

validation_sentences: 20000     # primeras N oraciones reservadas para validar
embedding_dim: 50
batch_size: 512
learning_rate: 0.003
epochs: 15
negative_samples: 10
seed: 42
```

`src/config.py` centraliza la carga y las rutas derivadas, de modo que las rutas
relativas del YAML se resuelven siempre contra la raíz del proyecto y el mismo
archivo funciona desde un notebook o desde la línea de comandos.

### Orden de escalado sugerido (una variable por vez)

1. `base_5k_50_2` — punto de partida
2. `5k_100_2` — efecto de la dimensión
3. `5k_100_5` — efecto del contexto
4. `10k_100_5` — efecto del vocabulario
5. `20k_200-300_10` — configuración final, si el cómputo lo permite

---

## Resultados medidos

### El corpus

Un pase completo con tokenización toma ~46 min (~35k oraciones/s):

| | |
|---|---|
| oraciones | 95.800.000 |
| tokens (tras tokenizar) | 2.028.209.650 |
| tokens por oración | 21,2 de promedio (p95 ≈ 62) |
| archivo | 2,97 GB comprimido |

Cobertura de tokens según `vocab_size`, medida sobre una muestra de 500k
oraciones con `min_count=5`:

| vocab_size | cobertura | `<UNK>` |
|---|---|---|
| 5.000 | 90,4% | 9,6% |
| 10.000 | 94,4% | 5,6% |
| 15.000 | 96,1% | 3,9% |
| 20.000 | 97,0% | 3,0% |

Sobre el **corpus completo** el vocabulario de 5.000 baja a **83,3% de cobertura
(16,7% `<UNK>`)**: la muestra sobreestima porque tiene mucha menos diversidad
léxica. Conviene tenerlo en cuenta al leer la grilla de arriba.

### El modelo

`src/model.py` implementa CBOW con **dos** matrices de embeddings —
`input_embeddings` (la palabra como contexto, lo que se exporta) y
`output_embeddings` (la palabra como target, se descarta)— y **negative sampling**
en la salida, con negativos sorteados según `frecuencia^0.75`.

Las `output_embeddings` se inicializan en **cero**, lo que hace que la loss del
primer paso valga exactamente `(1+K)·ln2 = 7,6246`: una verificación de sanidad
integrada.

Medido en la RTX 3060, con `B=512` y `K=10`, comparando ambos caminos completos
(hasta la loss, con `backward`):

| vocab | negative sampling | softmax completo | memoria NS | memoria softmax |
|---|---|---|---|---|
| 5.000 | 1,87 ms | **0,95 ms** | 46 MB | 75 MB |
| 20.000 | 1,91 ms | 2,34 ms | 57 MB | 176 MB |
| 50.000 | 1,94 ms | 5,93 ms | 81 MB | 374 MB |
| 200.000 | 2,64 ms | 23,22 ms | 196 MB | 1.368 MB |

Con vocabulario 5.000 el softmax completo es de hecho **más rápido**: un matmul
`(512,50)×(50,5000)` lo resuelve la GPU de un saque, mientras que el negative
sampling paga varios lanzamientos de kernel. El cruce está cerca de vocabulario
20.000 — justo el techo de la grilla del proyecto. La ventaja real a esta escala
es la **memoria**: el softmax materializa `(B, V)` más su gradiente, y esta placa
tiene 6 GB.

### Corrida piloto sobre una muestra

Entrenar sobre las 95,8M de oraciones cuesta horas por época. Para validar el
pipeline y tener una primera lectura de la calidad existe
`configs/piloto_5k_50_2.yaml`, que entrena sobre una muestra aleatoria de **2M de
oraciones** (~2% del corpus) materializada a un `.txt`.

Tres decisiones hacen que el piloto sea **comparable** con la corrida final:

- **`vocab_from: base_5k_50_2`** — reutiliza el vocabulario ya construido sobre el
  corpus completo. El piloto entrena en el mismo espacio de índices que la corrida
  definitiva, así que sus embeddings y sus métricas se comparan directamente. El
  CLI de `src/vocabulary.py` se niega a reconstruir un vocabulario prestado, para
  no pisar el de la corrida dueña.
- **Muestreo aleatorio, no un prefijo** — usar `limit=N` tomaría las primeras N
  oraciones, y el corpus arranca con texto coránico y actas del Parlamento
  Europeo: eso es una muestra del principio del archivo, no del idioma.
- **La muestra se baraja al escribirla** (por defecto; `--no-shuffle` lo
  desactiva). El SBWC está ordenado por fuente, y `CBOWIterableDataset` no puede
  barajar —no se puede barajar lo que todavía no se leyó—, así que sin esto el
  entrenamiento vería miles de oraciones de una misma fuente seguidas, con
  gradientes sesgados en cada tramo.

**La muestra, verificada.** 1.999.124 oraciones, 242 MB, generada en 6 min.
Comparada contra el corpus completo resulta ser una miniatura casi exacta:

| | Muestra (2M) | Corpus (95,8M) |
|---|---|---|
| tokens por oración | 21,16 | 21,17 |
| **cobertura del vocabulario** | **83,26%** | **83,26%** |
| top-15 de palabras | idéntico, en el mismo orden | — |

Como contraste, la muestra sesgada por prefijo (`limit=500_000`) que se usó para
explorar en la Fase 2 compartía solo el **69,3%** de las palabras con el
vocabulario real y reportaba 90,4% de cobertura en vez de 83,3%.

**Costo de una época:**

| | Piloto (2M) | Corpus completo (95,8M) |
|---|---|---|
| pares por época | 5.061.115 | ~242.500.000 |
| batches de 512 | 9.885 | ~473.700 |
| generar los pares | **52 s** | ~55-70 min |

Medido: 2,53 pares por oración, 97.629 pares/s. El cuello de botella es la **CPU**
generando pares (~87 s/época), no la GPU (~26 s/época).

### Entrenamiento

15 épocas sobre 2M de oraciones (75,2 millones de pares), 57 min en la RTX 3060:

| | Loss entrenamiento | Loss validación |
|---|---|---|
| antes del primer paso | 7,6246 | — |
| tras la época 0 | 2,9664 | 2,7528 |
| **tras la época 14** | **2,6358** | **2,6430** |

Casi todo el aprendizaje ocurre en la primera época; las 14 restantes son
refinamiento. Entrenamiento y validación van pegadas hasta el final (diferencia
final: 0,0072), así que no hay sobreajuste. Entre las épocas 10 y 14 la validación
baja 0,011: con 8-10 épocas se obtendría casi lo mismo.

### Calidad de los embeddings

| Palabra | Vecinos más cercanos |
|---|---|
| `lunes` | viernes, jueves, sábado, horario, horas, domingo |
| `febrero` | marzo (0,98), abril (0,98), octubre (0,98), mayo (0,97) |
| `tres` | cuatro (0,96), dos (0,95), cinco (0,92), siete (0,90) |
| `rojo` | azul (0,93), blanco (0,88), negro (0,87), amarillo (0,85) |
| `argentina` | argentino, republica, chile, paraguay, mendoza, uruguay |
| `presidente` | presidenta (0,85), vicepresidente (0,83), presidencia (0,79) |

Sin que nadie le dijera qué es un día de la semana, un mes o un color, el modelo
los agrupó a partir de una sola señal: qué palabras aparecen cerca de cuáles.

Precisión sobre el set fijo de analogías en español (respuesta correcta en primer
lugar) — esta es la métrica que permitirá comparar configuraciones con un número
en la Fase 7:

| Categoría | Precisión | |
|---|---|---|
| país → gentilicio | 76,7% | 23/30 |
| género | 61,1% | 55/90 |
| plural | 27,8% | 25/90 |
| adjetivo → adverbio | 20,0% | 4/20 |
| verbo: presente → pasado | 19,6% | 11/56 |
| **GLOBAL** | **41,3%** | 118/286 |

El perfil es informativo: las relaciones **temáticas** se resuelven bien y las
**morfológicas** no. Es consistente con `context_size=2` —una ventana chica
captura de qué se habla, no la sintaxis— y es una hipótesis directamente
comprobable subiendo el contexto a 5 o 10 en la Fase 7.

### Exportación

| Archivo | Tamaño |
|---|---|
| `embeddings/piloto_5k_50_2.txt` | 2,31 MB |
| `embeddings/piloto_5k_50_2.bin` | 0,99 MB |

Verificado con **gensim**: lee los dos formatos y reproduce exactamente los mismos
vecinos y las mismas analogías que nuestro código.

`<UNK>` **no se exporta** (el encabezado dice 4.999, no 5.000). Con
`drop_unknown=True` ese token nunca aparece como contexto ni como target, así que
su vector jamás recibió un gradiente: su norma quedó en 0,040, la de la
inicialización, contra una mediana de 3,201 del resto. Publicarlo sería publicar
ruido.

> Un detalle contraintuitivo que salió de medir las normas: las palabras **más**
> frecuentes tienen vectores más **cortos**. `de` (la número uno) tiene norma 1,41
> y `italia` (la número mil) tiene 2,83. `de` aparece en contextos tan variados que
> sus actualizaciones se cancelan entre sí; `italia` se ve siempre en el mismo
> ambiente y su vector se estira en una dirección clara.

---

## Decisiones de diseño

- **Todo por streaming.** El `.bz2` se lee línea a línea con generadores, tanto
  para contar frecuencias como para generar los pares de entrenamiento. En ningún
  momento se materializa el corpus ni la lista completa de pares (los ~242M de
  pares del corpus completo ocuparían ~6,3 GB en RAM).
- **Notebooks como pruebas, no como implementación.** Cada notebook ejercita *una*
  fase llamando a funciones de `src/`. La lógica vive en los módulos, que son
  reutilizables y testeables.
- **Se guarda el estado del optimizador.** Adam mantiene medias móviles de los
  gradientes; retomar sin ellas equivale a reiniciarlo a mitad de camino.
- **La validación es determinista**: muestreador de negativos con semilla fija en
  cada evaluación y sin `set_epoch` en su dataset. Si no, la curva de validación no
  sería comparable entre épocas. Las primeras `validation_sentences` oraciones se
  reservan y se saltean al entrenar, así ningún par aparece en los dos conjuntos.
- **Learning rate con decaimiento lineal** (como word2vec), con piso del 5% para
  que la última época siga aprendiendo.
- **`history.json` liviano**, separado de los checkpoints: permite comparar
  corridas sin cargar ningún modelo.
- **Padding + máscara** en los contextos: las ventanas cerca del borde de la
  oración son más cortas, y el promedio del contexto se hace enmascarado para no
  diluir el vector con posiciones de relleno.

## Lo que falta

Con la grilla montada, cada pregunta nueva es un YAML.

**Lo primero: entrenar skip-gram.** El código de la Fase 8 está probado pero no
hay ninguna corrida. `piloto_sg_5k_50_2` es el gemelo exacto del piloto CBOW y
responde la pregunta más grande que queda abierta: si la arquitectura importa
más o menos que la dimensión, que fue el factor dominante en la Fase 7.

Después, las cuatro que dejó abiertas la Fase 7, en orden de interés:

1. **`dim 200`, contexto 2.** La dimensión fue el factor dominante y no sabemos
   dónde deja de rendir. El salto 50→100 dio +12,9 pp; el de 100→200 dirá si la
   curva sigue subiendo o se aplana.
2. **`context_size: 1`.** Si las ventanas angostas favorecen la morfología, el
   experimento natural no es subir el contexto sino **bajarlo**. Es la predicción
   más directa que deja la Fase 7 y no se probó.
3. **`subsampling_threshold: 1e-4`.** Con el umbral actual `la` sobrevive el 1,6%
   de las veces: el subsampling está borrando justo los artículos que marcan
   género y número. Es un sospechoso concreto de la debilidad morfológica.
4. **Vocabulario 10k y el corpus completo.** Las 2M de oraciones fijaron el techo
   de todas estas corridas; recién con el corpus entero se sabe cuánto de la
   diferencia era falta de datos. `configs/base.yaml` hoy solo se usó para
   construir el vocabulario.
