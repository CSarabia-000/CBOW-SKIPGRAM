"""Fase 3 — Generación de pares y datasets para PyTorch (CBOW y skip-gram).

La ventana se desliza sobre cada oración ya traducida a índices del vocabulario.
Para una oración `a b c d e` con `context_size=2`, la ventana centrada en `c`
da `[a, b, d, e]`. Lo que cambia entre las dos arquitecturas es **en qué
dirección se lee esa misma ventana**:

* **CBOW** — un par por token: `([a, b, d, e], c)`. Se predice el centro desde
  todo su contexto junto.
* **Skip-gram** — un par por cada palabra del contexto: `([c], a)`, `([c], b)`,
  `([c], d)`, `([c], e)`. Se predice cada vecino por separado desde el centro.

De ahí sale la asimetría de costo: con `context_size=2` skip-gram produce hasta
4 pares donde CBOW produce 1, así que una época cuesta varias veces más. A
cambio cada par es una señal más nítida, que es la razón por la que skip-gram
suele rendir mejor con palabras poco frecuentes.

La ventana es **fija**, no sorteada por token como en el word2vec original.
CBOW ya se entrenó así, y mantener la misma ventana en las dos arquitecturas es
lo que deja que la comparación aísle la arquitectura y nada más.

Todo se produce por streaming: los generadores consumen el `.bz2` oración por
oración, sin materializar la lista completa de pares. Sobre el corpus entero esa
lista tendría miles de millones de elementos.

Dos envoltorios para el `DataLoader`, cada uno con su variante por arquitectura:

* `CBOWIterableDataset` / `SkipGramIterableDataset` — streaming, los que se usan
  con el corpus completo.
* `CBOWDataset` / `SkipGramDataset` — materializan una cantidad acotada de pares
  en arrays de numpy; sirven para notebooks, pruebas y configuraciones chicas,
  donde tener `len()` y barajado real es cómodo.

`build_dataset_from_config` elige la variante leyendo `arch` del YAML, para que
`src.train` no tenga que saber qué arquitectura está entrenando.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from src import configure_console
from src.config import arch as config_arch
from src.config import corpus_path, load_config, vocab_path
from src.corpus import stream_tokens
from src.vocabulary import Vocabulary

__all__ = [
    "generate_pairs",
    "generate_skipgram_pairs",
    "pair_generator",
    "input_width",
    "pad_context",
    "format_pair",
    "PairDataset",
    "CBOWDataset",
    "SkipGramDataset",
    "PairIterableDataset",
    "CBOWIterableDataset",
    "SkipGramIterableDataset",
    "build_dataset_from_config",
    "build_map_dataset_from_config",
]


def generate_pairs(
    path: str | Path,
    vocab: Vocabulary,
    context_size: int,
    sample_fraction: float = 1.0,
    *,
    limit: int | None = None,
    seed: int = 42,
    drop_unknown: bool = True,
    subsampling_threshold: float = 0.0,
    skip_sentences: int = 0,
    shard_index: int = 0,
    shard_count: int = 1,
    **tokenizer_kwargs,
) -> Iterator[tuple[list[int], int]]:
    """Genera pares `(contexto, target)` como índices del vocabulario.

    Recorre el corpus por streaming, tokeniza cada oración, la traduce a índices
    y desliza una ventana de `context_size` palabras a cada lado.

    En los bordes de la oración el contexto queda **más corto** (el primer token
    no tiene nada a la izquierda). No se rellena acá: el contexto se entrega con
    su largo real y el relleno ocurre recién en la capa de tensores, que además
    produce la máscara. Así no se pierden los targets de los extremos, que con
    `context_size=10` serían la mayoría de las oraciones.

    Args:
        path: ruta al corpus.
        vocab: vocabulario de la Fase 2.
        context_size: palabras de contexto a cada lado del target.
        sample_fraction: fracción de líneas del corpus a considerar.
        limit: cortar tras N oraciones leídas.
        seed: semilla del muestreo de líneas y del subsampling.
        drop_unknown: si las palabras fuera de vocabulario se eliminan de la
            oración (por defecto) en vez de convertirse en `<UNK>`. Eliminarlas
            evita que el modelo gaste capacidad prediciendo un token comodín que
            no significa nada; el costo es que dos palabras separadas por un OOV
            quedan adyacentes.
        subsampling_threshold: umbral `t` de Mikolov. Con `> 0` cada ocurrencia
            se descarta al azar según `Vocabulary.keep_probabilities`. Con `0`
            no se aplica subsampling.
        skip_sentences: descartar las primeras N oraciones. Sirve para separar
            validación de entrenamiento sin duplicar archivos: la validación usa
            `limit=N` y el entrenamiento `skip_sentences=N`.
        shard_index, shard_count: reparto entre procesos. Cada shard se queda
            con las oraciones cuyo índice le corresponde, de modo que la unión
            es el corpus completo sin repeticiones. Se reparte por **oración**,
            no por par, para que cada worker tokenice solo su parte: repartir
            por par obligaría a todos a tokenizar todo y tirar lo ajeno, o sea
            paralelismo sin ninguna ganancia.
        Los demás argumentos se pasan al tokenizador.

    Yields:
        `(contexto, target)`, donde `contexto` es una lista de entre 1 y
        `2 * context_size` índices, y `target` es un índice.
    """
    if context_size < 1:
        raise ValueError(f"context_size debe ser >= 1, no {context_size!r}")
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError(f"shard inválido: {shard_index}/{shard_count}")

    keep_probabilities: list[float] | None = None
    rng: random.Random | None = None
    if subsampling_threshold > 0:
        keep_probabilities = vocab.keep_probabilities(subsampling_threshold)
        # Cada shard sortea con su propia semilla: si compartieran el flujo de
        # azar tomarían decisiones acopladas sobre oraciones distintas.
        rng = random.Random(seed + 1000 * shard_index)

    sentence_limit = None if limit is None else limit + skip_sentences
    for sentence_index, tokens in enumerate(
        stream_tokens(path, sample_fraction, limit=sentence_limit, seed=seed, **tokenizer_kwargs)
    ):
        if sentence_index < skip_sentences:
            continue
        if shard_count > 1 and (sentence_index - skip_sentences) % shard_count != shard_index:
            continue

        indices = vocab.encode(tokens, drop_unknown=drop_unknown)

        if keep_probabilities is not None:
            indices = [i for i in indices if rng.random() < keep_probabilities[i]]

        length = len(indices)
        if length < 2:
            # Con un solo token no hay contexto posible.
            continue

        for position, target in enumerate(indices):
            start = max(0, position - context_size)
            end = min(length, position + context_size + 1)
            context = indices[start:position] + indices[position + 1 : end]
            if context:
                yield context, target


def generate_skipgram_pairs(
    path: str | Path,
    vocab: Vocabulary,
    context_size: int,
    sample_fraction: float = 1.0,
    **kwargs,
) -> Iterator[tuple[list[int], int]]:
    """Genera pares `([centro], vecino)`: la misma ventana, leída al revés.

    Se apoya en `generate_pairs` en vez de recorrer el corpus por su cuenta, y
    eso es deliberado: el muestreo de líneas, el subsampling, `drop_unknown`, el
    corte de validación y el reparto entre workers ya están resueltos y probados
    ahí. Duplicar esa lógica para skip-gram sería duplicar también sus bugs, y
    —peor para la comparación— abriría la puerta a que las dos arquitecturas
    terminen entrenando sobre ventanas sutilmente distintas.

    Cada par de CBOW `([a, b, d, e], c)` se **expande** en un par por vecino:
    `([c], a)`, `([c], b)`, `([c], d)`, `([c], e)`.

    El centro se entrega como lista de un elemento, no como entero suelto, para
    que el contrato del batch sea el mismo en las dos arquitecturas
    (`(entrada, máscara, target)`) y `src.train` no tenga que ramificar.

    Yields:
        `([centro], vecino)` — la lista siempre tiene largo 1.
    """
    for context, target in generate_pairs(
        path, vocab, context_size, sample_fraction, **kwargs
    ):
        centro = [target]
        for vecino in context:
            yield centro, vecino


def pair_generator(arch: str):
    """El generador de pares que le corresponde a una arquitectura."""
    if arch == "skipgram":
        return generate_skipgram_pairs
    if arch == "cbow":
        return generate_pairs
    raise ValueError(f"arquitectura desconocida: {arch!r}")


def _check_arch(cls: type, config: dict) -> None:
    """Falla si la clase de dataset no corresponde al `arch` de la corrida.

    Existe porque el error contrario es silencioso: entrenar CBOW sobre pares de
    skip-gram (o al revés) produce curvas de loss perfectamente creíbles, y el
    problema recién se nota al comparar las dos corridas, con horas de cómputo
    ya gastadas.
    """
    declarada = config_arch(config)
    if declarada != cls.ARCH:
        raise ValueError(
            f"{cls.__name__} no sirve para la corrida {config.get('name')!r}, "
            f"que declara arch={declarada!r}. Usá build_dataset_from_config()."
        )


def input_width(arch: str, context_size: int) -> int:
    """Ancho de la entrada de cada ejemplo, en índices.

    CBOW entrega hasta `2 * context_size` palabras de contexto; skip-gram
    entrega una sola palabra (el centro), así que su relleno es siempre trivial
    y su máscara, todo unos.
    """
    return 1 if arch == "skipgram" else 2 * context_size


def pad_context(
    context: Sequence[int],
    context_size: int,
    pad_index: int = 0,
    *,
    width: int | None = None,
) -> tuple[list[int], list[int]]:
    """Rellena un contexto hasta `width` (por defecto `2 * context_size`).

    La máscara vale 1 en las posiciones reales y 0 en el relleno, para que el
    modelo pueda promediar solo sobre las primeras. El valor con el que se
    rellena es irrelevante justamente porque la máscara lo anula.

    En skip-gram la entrada mide 1 y nunca hay relleno; la función se aplica
    igual para que las dos arquitecturas produzcan batches de la misma forma.

    Returns:
        `(indices, mascara)`, ambos de largo `width`.
    """
    width = 2 * context_size if width is None else width
    indices = list(context[:width])
    mask = [1] * len(indices)
    missing = width - len(indices)
    if missing:
        indices.extend([pad_index] * missing)
        mask.extend([0] * missing)
    return indices, mask


def format_pair(
    context: Sequence[int], target: int, vocab: Vocabulary, arch: str = "cbow"
) -> str:
    """Representación legible de un par, con las palabras en vez de los índices.

    La flecha apunta siempre de la entrada a lo que se predice, así que en
    skip-gram se lee `centro -> [vecino]` y en CBOW `[contexto] -> centro`.
    """
    palabras = " ".join(vocab.word(i) for i in context)
    if arch == "skipgram":
        return f"{palabras}  ->  [{vocab.word(target)}]"
    return f"[{palabras}]  ->  {vocab.word(target)}"


class PairDataset(Dataset):
    """Dataset map-style con los pares materializados en arrays de numpy.

    Clase base: no se instancia directamente, se usa `CBOWDataset` o
    `SkipGramDataset`, que solo fijan `ARCH`. Todo lo demás —el relleno, los
    arrays, el `__getitem__`— es común a las dos arquitecturas, porque lo único
    que cambia entre ellas es qué generador produce los pares y cuánto mide la
    entrada.

    Guardar los pares como `int32` en vez de listas de Python los hace unas 10
    veces más compactos, pero aun así **no** entran los del corpus completo: es
    para notebooks, pruebas y configuraciones chicas. Para entrenar sobre todo
    el corpus se usa la variante iterable.

    Cada elemento es `(entrada, mascara, target)` con `entrada` y `mascara` como
    arrays de numpy de largo `input_width(ARCH, context_size)` y `target` como
    entero. El `DataLoader` los apila en tensores `(B, W)`, `(B, W)` y `(B,)`.
    """

    #: Arquitectura que fija la subclase. Vacía en la base, que es abstracta.
    ARCH: str = ""

    def __init__(
        self,
        pairs: Iterable[tuple[Sequence[int], int]],
        context_size: int,
        *,
        max_pairs: int | None = None,
        pad_index: int = 0,
    ) -> None:
        if not self.ARCH:
            raise TypeError(
                "PairDataset es abstracta: usá CBOWDataset o SkipGramDataset"
            )
        self.context_size = context_size
        self.pad_index = pad_index
        self.width = input_width(self.ARCH, context_size)

        contexts: list[list[int]] = []
        masks: list[list[int]] = []
        targets: list[int] = []
        for count, (context, target) in enumerate(pairs):
            if max_pairs is not None and count >= max_pairs:
                break
            indices, mask = pad_context(
                context, context_size, pad_index, width=self.width
            )
            contexts.append(indices)
            masks.append(mask)
            targets.append(target)

        self.contexts = np.asarray(contexts, dtype=np.int32).reshape(-1, self.width)
        self.masks = np.asarray(masks, dtype=np.int8).reshape(-1, self.width)
        self.targets = np.asarray(targets, dtype=np.int32).reshape(-1)

    @classmethod
    def from_config(
        cls,
        config: dict,
        vocab: Vocabulary,
        *,
        max_pairs: int | None = None,
        limit: int | None = None,
    ) -> "PairDataset":
        """Construye el dataset tomando los parámetros de un YAML de `configs/`.

        Si la configuración declara otra arquitectura, falla en vez de generar
        pares con la forma equivocada: una corrida entrenada sobre los pares de
        la arquitectura que no era se ve perfectamente normal en las curvas y
        solo se nota mucho después, al comparar.
        """
        _check_arch(cls, config)
        pairs = cls._pairs(
            corpus_path(config),
            vocab,
            config["context_size"],
            config.get("corpus_sample_fraction", 1.0),
            limit=limit,
            seed=config.get("seed", 42),
            drop_unknown=config.get("drop_unknown", True),
            subsampling_threshold=config.get("subsampling_threshold", 0.0),
            **config.get("tokenizer", {}),
        )
        return cls(pairs, config["context_size"], max_pairs=max_pairs)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, int]:
        """Devuelve arrays de numpy; el `collate` del DataLoader los apila.

        Igual que en `CBOWIterableDataset`: convertir a tensor de a un elemento
        es notablemente más caro que dejar que el DataLoader lo haga por batch.
        """
        if index >= len(self.targets):
            raise IndexError(index)
        return (
            self.contexts[index].astype(np.int64),
            self.masks[index].astype(np.float32),
            int(self.targets[index]),
        )

    @classmethod
    def _pairs(cls, *args, **kwargs) -> Iterator[tuple[list[int], int]]:
        return pair_generator(cls.ARCH)(*args, **kwargs)

    @property
    def nbytes(self) -> int:
        """Memoria ocupada por los pares, en bytes."""
        return self.contexts.nbytes + self.masks.nbytes + self.targets.nbytes

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(pairs={len(self):,}, "
            f"context_size={self.context_size}, "
            f"memoria={self.nbytes / 1024**2:.1f} MB)"
        )


class CBOWDataset(PairDataset):
    """Pares `([contexto], centro)` materializados. Ver `PairDataset`."""

    ARCH = "cbow"


class SkipGramDataset(PairDataset):
    """Pares `([centro], vecino)` materializados. Ver `PairDataset`.

    Ojo con `max_pairs` en los notebooks: para un mismo número de oraciones
    leídas, skip-gram produce hasta `2 * context_size` veces más pares que CBOW,
    así que el mismo tope corta muchísimo antes en el corpus.
    """

    ARCH = "skipgram"


class PairIterableDataset(IterableDataset):
    """Dataset de streaming: recorre el corpus y emite pares sin acumularlos.

    Clase base de `CBOWIterableDataset` y `SkipGramIterableDataset`, que solo
    fijan `ARCH`.

    Es el que se usa para entrenar sobre el corpus completo, donde la lista de
    pares no entra en memoria. No tiene `len()` ni permite barajado global; el
    `DataLoader` debe usarse con `shuffle=False` (opcionalmente con un
    `ShufflerIterDataPipe` o un buffer de barajado si hiciera falta).

    Con `num_workers > 0` el corpus se reparte **por oración**: cada worker
    tokeniza solo las oraciones que le tocan, así que el trabajo se divide de
    verdad. En skip-gram el reparto sigue siendo por oración y no por par, así
    que los pares expandidos de una misma ventana caen todos en el mismo worker;
    no importa, porque son ejemplos independientes. Todos leen el archivo entero (el I/O se repite), pero lo caro es
    tokenizar, y eso sí se paraleliza. Sobre el `.txt` de la muestra conviene
    `num_workers` > 0; sobre el `.bz2` de 3 GB la descompresión repetida puede
    comerse la ganancia.
    """

    def __init__(
        self,
        path: str | Path,
        vocab: Vocabulary,
        context_size: int,
        sample_fraction: float = 1.0,
        *,
        limit: int | None = None,
        seed: int = 42,
        drop_unknown: bool = True,
        subsampling_threshold: float = 0.0,
        skip_sentences: int = 0,
        pad_index: int = 0,
        **tokenizer_kwargs,
    ) -> None:
        if not self.ARCH:
            raise TypeError(
                "PairIterableDataset es abstracta: usá CBOWIterableDataset o "
                "SkipGramIterableDataset"
            )
        self.path = path
        self.vocab = vocab
        self.context_size = context_size
        self.width = input_width(self.ARCH, context_size)
        self.sample_fraction = sample_fraction
        self.limit = limit
        self.seed = seed
        self.drop_unknown = drop_unknown
        self.subsampling_threshold = subsampling_threshold
        self.skip_sentences = skip_sentences
        self.pad_index = pad_index
        self.tokenizer_kwargs = tokenizer_kwargs
        self._epoch = 0

    @classmethod
    def from_config(
        cls,
        config: dict,
        vocab: Vocabulary,
        *,
        limit: int | None = None,
        skip_sentences: int = 0,
    ) -> "PairIterableDataset":
        """Construye el dataset tomando los parámetros de un YAML de `configs/`.

        Falla si la configuración declara otra arquitectura; ver
        `PairDataset.from_config`.
        """
        _check_arch(cls, config)
        return cls(
            corpus_path(config),
            vocab,
            config["context_size"],
            config.get("corpus_sample_fraction", 1.0),
            limit=limit,
            seed=config.get("seed", 42),
            drop_unknown=config.get("drop_unknown", True),
            subsampling_threshold=config.get("subsampling_threshold", 0.0),
            skip_sentences=skip_sentences,
            **config.get("tokenizer", {}),
        )

    def set_epoch(self, epoch: int) -> None:
        """Cambia la semilla efectiva para que cada época vea un sorteo distinto.

        Sin esto, el subsampling descartaría exactamente las mismas ocurrencias
        en todas las épocas y el modelo vería siempre los mismos pares.
        """
        self._epoch = epoch

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        worker = get_worker_info()
        num_workers = worker.num_workers if worker is not None else 1
        worker_id = worker.id if worker is not None else 0

        pairs = pair_generator(self.ARCH)(
            self.path,
            self.vocab,
            self.context_size,
            self.sample_fraction,
            limit=self.limit,
            seed=self.seed + self._epoch,
            drop_unknown=self.drop_unknown,
            subsampling_threshold=self.subsampling_threshold,
            skip_sentences=self.skip_sentences,
            shard_index=worker_id,
            shard_count=num_workers,
            **self.tokenizer_kwargs,
        )

        for context, target in pairs:
            indices, mask = pad_context(
                context, self.context_size, self.pad_index, width=self.width
            )
            # Se entregan arrays de numpy, no tensores. El `collate` por defecto
            # del DataLoader los convierte de a un batch entero, que es mucho más
            # barato que construir tres tensores diminutos por cada par: medido,
            # crear los tensores de a uno cuesta la mitad del rendimiento del
            # pipeline (93k -> 46k pares/s), más que generar los pares.
            yield (
                np.asarray(indices, dtype=np.int64),
                np.asarray(mask, dtype=np.float32),
                target,
            )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(context_size={self.context_size}, "
            f"subsampling={self.subsampling_threshold}, "
            f"drop_unknown={self.drop_unknown})"
        )


class CBOWIterableDataset(PairIterableDataset):
    """Streaming de pares `([contexto], centro)`. Ver `PairIterableDataset`."""

    ARCH = "cbow"


class SkipGramIterableDataset(PairIterableDataset):
    """Streaming de pares `([centro], vecino)`. Ver `PairIterableDataset`."""

    ARCH = "skipgram"


#: Qué clase le toca a cada arquitectura. `build_dataset_from_config` la usa.
_ITERABLE_POR_ARCH = {
    "cbow": CBOWIterableDataset,
    "skipgram": SkipGramIterableDataset,
}

_MAP_POR_ARCH = {
    "cbow": CBOWDataset,
    "skipgram": SkipGramDataset,
}


def build_dataset_from_config(
    config: dict,
    vocab: Vocabulary,
    *,
    limit: int | None = None,
    skip_sentences: int = 0,
) -> PairIterableDataset:
    """Dataset de streaming de la arquitectura que declare el YAML.

    Es el único punto del pipeline de entrenamiento que sabe que existe más de
    una arquitectura: `src.train` llama a esto y no vuelve a ramificar.
    """
    cls = _ITERABLE_POR_ARCH[config_arch(config)]
    return cls.from_config(config, vocab, limit=limit, skip_sentences=skip_sentences)


def build_map_dataset_from_config(
    config: dict,
    vocab: Vocabulary,
    *,
    max_pairs: int | None = None,
    limit: int | None = None,
) -> PairDataset:
    """Igual que `build_dataset_from_config`, pero materializando los pares."""
    cls = _MAP_POR_ARCH[config_arch(config)]
    return cls.from_config(config, vocab, max_pairs=max_pairs, limit=limit)


def _main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(
        description="Genera pares de la arquitectura que declare el YAML."
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--limit", type=int, default=2000, help="oraciones a leer")
    parser.add_argument("--show", type=int, default=10, help="pares a mostrar")
    args = parser.parse_args()

    config = load_config(args.config)
    vocab = Vocabulary.load(vocab_path(config))
    arquitectura = config_arch(config)
    print(
        f"config: {config['name']} | arch={arquitectura} | "
        f"context_size={config['context_size']} | {vocab}"
    )

    pairs = pair_generator(arquitectura)(
        corpus_path(config),
        vocab,
        config["context_size"],
        limit=args.limit,
        seed=config.get("seed", 42),
        drop_unknown=config.get("drop_unknown", True),
        subsampling_threshold=config.get("subsampling_threshold", 0.0),
        **config.get("tokenizer", {}),
    )

    total = 0
    for index, (context, target) in enumerate(pairs):
        if index < args.show:
            print(f"  {format_pair(context, target, vocab, arquitectura)}")
        total += 1

    print(f"\npares generados a partir de {args.limit:,} oraciones: {total:,}")


if __name__ == "__main__":
    _main()
