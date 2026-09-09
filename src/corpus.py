"""Fase 1 — Lectura en streaming del corpus SBWC y tokenización.

El corpus (`sbwce.clean.txt.bz2`, ~3 GB comprimido, una oración por línea) nunca
se descomprime a disco ni se carga entero en memoria: todo lo que expone este
módulo son generadores que consumen el `.bz2` línea a línea.
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import IO, Iterator

from src import configure_console

__all__ = [
    "NUM_TOKEN",
    "open_corpus",
    "stream_sentences",
    "tokenize",
    "stream_tokens",
    "write_sample",
    "count_lines",
    "corpus_stats",
]

#: Token con el que se normaliza cualquier secuencia de dígitos.
NUM_TOKEN = "<NUM>"

# Un token es una secuencia de letras (unicode, así entran acentos y "ñ") que
# puede llevar apóstrofe o guion interno ("o'donnell", "franco-alemán"). Las
# secuencias de dígitos se capturan aparte para poder normalizarlas a NUM_TOKEN.
_TOKEN_RE = re.compile(r"[^\W\d_]+(?:['\-][^\W\d_]+)*|\d+", re.UNICODE)

# El corpus trae guiones suaves incrustados dentro de palabras ("ré\xadgimen");
# hay que borrarlos antes de tokenizar o parten el token en dos. El apóstrofe
# tipográfico se unifica con el ASCII para no duplicar tipos ("o’donnell").
_SOFT_HYPHEN = "­"
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "´": "'"})


def open_corpus(path: str | Path, encoding: str = "utf-8") -> IO[str]:
    """Abre el corpus en modo texto según su extensión (.bz2, .gz o plano).

    Los bytes inválidos se reemplazan en vez de romper la lectura: sobre un
    corpus de miles de millones de tokens no queremos que una línea corrupta
    aborte un entrenamiento de horas.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".bz2":
        return bz2.open(path, "rt", encoding=encoding, errors="replace")
    if suffix == ".gz":
        return gzip.open(path, "rt", encoding=encoding, errors="replace")
    return path.open("rt", encoding=encoding, errors="replace")


def stream_sentences(
    path: str | Path,
    sample_fraction: float = 1.0,
    *,
    limit: int | None = None,
    seed: int = 42,
) -> Iterator[str]:
    """Genera las oraciones del corpus, una por línea, sin cargarlo en memoria.

    Args:
        path: ruta al corpus (`.bz2`, `.gz` o `.txt`).
        sample_fraction: probabilidad de conservar cada línea. Con 1.0 se
            recorre el corpus completo; con valores menores se obtiene una
            muestra aleatoria reproducible (depende solo de `seed`), útil para
            iterar rápido sin recorrer los 3 GB.
        limit: corta el generador tras producir esa cantidad de oraciones.
        seed: semilla del muestreo.

    Yields:
        Cada oración ya sin salto de línea ni espacios sobrantes. Las líneas
        vacías se descartan.
    """
    if not 0.0 < sample_fraction <= 1.0:
        raise ValueError(f"sample_fraction debe estar en (0, 1], no {sample_fraction!r}")
    if limit is not None and limit <= 0:
        raise ValueError(f"limit debe ser positivo o None, no {limit!r}")

    rng = random.Random(seed)
    produced = 0
    with open_corpus(path) as handle:
        for line in handle:
            if sample_fraction < 1.0 and rng.random() >= sample_fraction:
                continue
            sentence = line.strip()
            if not sentence:
                continue
            yield sentence
            produced += 1
            if limit is not None and produced >= limit:
                return


def tokenize(
    sentence: str,
    *,
    lowercase: bool = True,
    number_token: str | None = NUM_TOKEN,
    min_length: int = 1,
) -> list[str]:
    """Convierte una oración en una lista de tokens.

    Normaliza a NFC, elimina guiones suaves, unifica apóstrofes, pasa a
    minúsculas y se queda solo con secuencias de letras; la puntuación residual
    del corpus (`.`, `/`, `,`, paréntesis) actúa como separador y se descarta.

    Las secuencias de dígitos se colapsan en un único `number_token`: "3.5" o
    "12/03/1998" aportan un solo token y no ocupan varias posiciones de la
    ventana de contexto.

    Args:
        sentence: oración cruda.
        lowercase: si se pasa todo a minúsculas.
        number_token: token con el que se reemplaza cada número.
            Con `None` los números se descartan.
        min_length: longitud mínima de un token alfabético para conservarlo.

    Returns:
        Lista de tokens (posiblemente vacía).
    """
    text = unicodedata.normalize("NFC", sentence).translate(_APOSTROPHES)
    text = text.replace(_SOFT_HYPHEN, "")
    if lowercase:
        text = text.lower()

    tokens: list[str] = []                                                        
    previous_was_number = False
    for match in _TOKEN_RE.findall(text):
        is_number = match[0].isdigit()
        if is_number:
            if number_token is not None and not previous_was_number:
                tokens.append(number_token)
        elif len(match) >= min_length:
            tokens.append(match)
        previous_was_number = is_number
    return tokens


def stream_tokens(
    path: str | Path,
    sample_fraction: float = 1.0,
    *,
    limit: int | None = None,
    seed: int = 42,
    **tokenizer_kwargs,
) -> Iterator[list[str]]:
    """Igual que `stream_sentences` pero entregando cada oración ya tokenizada.

    Es el generador que consumirán las fases siguientes (vocabulario y pares
    contexto-target). Las oraciones que quedan sin ningún token se omiten. Los
    `tokenizer_kwargs` se pasan tal cual a `tokenize`.
    """
    for sentence in stream_sentences(path, sample_fraction, limit=limit, seed=seed):
        tokens = tokenize(sentence, **tokenizer_kwargs)
        if tokens:
            yield tokens


def write_sample(
    source: str | Path,
    destination: str | Path,
    sample_fraction: float,
    *,
    seed: int = 42,
    limit: int | None = None,
    shuffle: bool = False,
    progress: bool = False,
) -> int:
    """Escribe a `destination` una muestra aleatoria de oraciones del corpus.

    Materializar la muestra una sola vez resuelve dos cosas a la vez:

    * **Evita el sesgo de `limit`**, que toma las primeras N líneas y por lo
      tanto una muestra del principio del archivo, no del idioma.
    * **Elimina el costo por época.** Descomprimir los 3 GB cuesta unos minutos
      cada vez que se recorre el corpus; sobre el archivo de muestra, en texto
      plano y mucho más chico, las corridas siguientes leen a velocidad de disco.

    Args:
        source: corpus de entrada.
        destination: archivo de salida (`.txt`, o `.bz2`/`.gz` si se prefiere
            comprimirlo a costa de volver a pagar descompresión al leerlo).
        sample_fraction: probabilidad de conservar cada oración.
        seed: semilla del muestreo, para que sea reproducible.
        limit: cortar tras escribir esa cantidad de oraciones.
        shuffle: barajar las oraciones antes de escribirlas. **Muy
            recomendable.** Sin esto la muestra conserva el orden del corpus, y
            como el SBWC está ordenado por fuente (arranca con texto coránico,
            sigue con actas parlamentarias), eso trae dos problemas:

            1. Cualquier *prefijo* de la muestra sigue estando sesgado, así que
               `limit` vuelve a ser engañoso sobre el archivo muestreado.
            2. `CBOWIterableDataset` no puede barajar —no se puede barajar lo
               que todavía no se leyó—, así que el entrenamiento vería todas las
               oraciones de una misma fuente seguidas, que es malo para SGD.

            El costo es que las oraciones se acumulan en memoria antes de
            escribirse (unos 350 MB para 2M de oraciones).
        progress: barra `tqdm` si está instalado.

    Returns:
        Cantidad de oraciones escritas.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    sentences: Iterator[str] = stream_sentences(
        source, sample_fraction, limit=limit, seed=seed
    )
    if progress:
        try:
            from tqdm.auto import tqdm

            sentences = tqdm(sentences, unit=" oraciones", unit_scale=True)
        except ImportError:
            pass

    if shuffle:
        collected = list(sentences)
        random.Random(seed).shuffle(collected)
        sentences = iter(collected)

    written = 0
    suffix = destination.suffix.lower()
    if suffix == ".bz2":
        handle = bz2.open(destination, "wt", encoding="utf-8")
    elif suffix == ".gz":
        handle = gzip.open(destination, "wt", encoding="utf-8")
    else:
        handle = destination.open("wt", encoding="utf-8", newline="\n")

    with handle:
        for sentence in sentences:
            handle.write(sentence)
            handle.write("\n")
            written += 1
    return written


def count_lines(path: str | Path, *, progress: bool = False) -> int:
    """Cuenta las líneas del corpus por streaming (reproducible desde Python).

    Sobre el corpus completo tarda del orden de minutos, porque hay que
    descomprimir los 3 GB. Con `progress=True` muestra una barra `tqdm` si está
    instalado.
    """
    handle = open_corpus(path)
    try:
        iterator: Iterator[str] = handle
        if progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(handle, unit=" líneas", unit_scale=True)
            except ImportError:
                pass
        return sum(1 for _ in iterator)
    finally:
        handle.close()


def corpus_stats(
    path: str | Path,
    sample_fraction: float = 1.0,
    *,
    limit: int | None = None,
    seed: int = 42,
    track_vocabulary: bool = True,
    **tokenizer_kwargs,
) -> dict:
    """Recorre el corpus (o una muestra) y devuelve estadísticas descriptivas.

    Args:
        track_vocabulary: si se acumula el conjunto de tipos distintos. Sobre el
            corpus completo eso ocupa memoria significativa, así que conviene
            dejarlo activo solo para muestras (que es su uso en el notebook 01);
            el vocabulario "de verdad" se construye en la Fase 2.

    Returns:
        Diccionario con número de oraciones, tokens, tipos distintos, promedio
        de tokens por oración, longitud mínima/máxima y el histograma completo
        de longitudes (`Counter` longitud -> cantidad de oraciones).
    """
    n_sentences = 0
    n_tokens = 0
    min_len: int | None = None
    max_len = 0
    lengths: Counter[int] = Counter()
    types: set[str] = set()

    for tokens in stream_tokens(
        path, sample_fraction, limit=limit, seed=seed, **tokenizer_kwargs
    ):
        length = len(tokens)
        n_sentences += 1
        n_tokens += length
        lengths[length] += 1
        max_len = max(max_len, length)
        min_len = length if min_len is None else min(min_len, length)
        if track_vocabulary:
            types.update(tokens)

    return {
        "n_sentences": n_sentences,
        "n_tokens": n_tokens,
        "n_types": len(types) if track_vocabulary else None,
        "avg_tokens_per_sentence": n_tokens / n_sentences if n_sentences else 0.0,
        "min_sentence_length": min_len or 0,
        "max_sentence_length": max_len,
        "length_histogram": lengths,
    }


def _main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(description="Exploración del corpus SBWC.")
    parser.add_argument("--path", default="data/raw/sbwce.clean.txt.bz2")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument(
        "--count-lines",
        action="store_true",
        help="solo cuenta las líneas del corpus (recorre el archivo completo)",
    )
    parser.add_argument(
        "--sample-to",
        default=None,
        help="escribe una muestra aleatoria a este archivo y termina",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="no barajar la muestra (conserva el orden del corpus)",
    )
    args = parser.parse_args()

    if args.sample_to:
        written = write_sample(
            args.path,
            args.sample_to,
            args.sample_fraction,
            seed=args.seed,
            limit=args.limit,
            shuffle=not args.no_shuffle,
            progress=True,
        )
        destination = Path(args.sample_to)
        print(f"oraciones escritas: {written:,}")
        print(f"archivo           : {destination} "
              f"({destination.stat().st_size / 1024**2:,.1f} MB)")
        return

    if args.count_lines:
        print(f"líneas: {count_lines(args.path, progress=True):,}")
        return

    stats = corpus_stats(args.path, args.sample_fraction, limit=args.limit)
    histogram = stats.pop("length_histogram")
    for key, value in stats.items():
        print(f"{key}: {value:,.2f}" if isinstance(value, float) else f"{key}: {value}")
    print(f"longitudes distintas: {len(histogram):,}")


if __name__ == "__main__":
    _main()
