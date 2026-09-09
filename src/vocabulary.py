"""Fase 2 — Construcción del vocabulario y subsampling de palabras frecuentes.

El conteo de frecuencias se hace por streaming sobre el corpus comprimido (ver
`src.corpus`): en ningún momento se materializa el texto. Lo único que vive en
memoria es el `Counter` de tipos distintos, cuyo tamaño se puede acotar con
`max_types` para corpus muy grandes.

El vocabulario resultante se serializa a `data/processed/{config_name}/vocab.json`
con el mapeo palabra→índice e índice→palabra.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src import configure_console
from src.config import (
    corpus_path,
    load_config,
    reuses_vocabulary,
    vocab_owner,
    vocab_path,
)
from src.corpus import stream_tokens

__all__ = [
    "UNK_TOKEN",
    "Vocabulary",
    "count_word_frequencies",
    "build_vocabulary",
    "build_vocabulary_from_config",
    "subsample",
]

#: Token que absorbe todas las palabras fuera del vocabulario. Ocupa el índice 0.
#: El tokenizador nunca puede producirlo (descarta `<` y `>`), así que no colisiona.
UNK_TOKEN = "<UNK>"


@dataclass
class Vocabulary:
    """Mapeo palabra↔índice más las frecuencias con las que se construyó.

    El índice 0 es siempre `unk_token`; el resto de las palabras van ordenadas
    por frecuencia descendente (desempatando alfabéticamente, para que la
    construcción sea determinista). `counts[0]` guarda cuántas ocurrencias del
    corpus cayeron en `<UNK>`, lo que permite calcular la cobertura.

    Attributes:
        itos: índice → palabra.
        counts: índice → frecuencia en el corpus.
        total_tokens: total de tokens vistos al construirlo (incluye los OOV).
        unk_token: literal usado para las palabras fuera de vocabulario.
        min_count: frecuencia mínima exigida al construirlo (informativo).
    """

    itos: list[str]
    counts: list[int]
    total_tokens: int
    unk_token: str = UNK_TOKEN
    min_count: int = 1
    stoi: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.itos) != len(self.counts):
            raise ValueError("itos y counts deben tener la misma longitud")
        if not self.itos or self.itos[0] != self.unk_token:
            raise ValueError(f"el índice 0 debe ser {self.unk_token!r}")
        self.stoi = {word: index for index, word in enumerate(self.itos)}

    # -- acceso básico ---------------------------------------------------

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, word: str) -> bool:
        return word in self.stoi

    def __repr__(self) -> str:
        return (
            f"Vocabulary(size={len(self)}, total_tokens={self.total_tokens:,}, "
            f"coverage={self.coverage:.4f}, min_count={self.min_count})"
        )

    @property
    def unk_index(self) -> int:
        return 0

    def index(self, word: str) -> int:
        """Índice de `word`, o el de `<UNK>` si está fuera del vocabulario."""
        return self.stoi.get(word, self.unk_index)

    def word(self, index: int) -> str:
        return self.itos[index]

    def encode(self, tokens: Iterable[str], *, drop_unknown: bool = False) -> list[int]:
        """Traduce una lista de tokens a índices.

        Con `drop_unknown=True` las palabras fuera de vocabulario se descartan en
        vez de mapearse a `<UNK>`.
        """
        if drop_unknown:
            return [self.stoi[t] for t in tokens if t in self.stoi]
        return [self.stoi.get(t, self.unk_index) for t in tokens]

    def decode(self, indices: Iterable[int]) -> list[str]:
        return [self.itos[i] for i in indices]

    def most_common(self, n: int = 20, *, include_unk: bool = False) -> list[tuple[str, int]]:
        """Las `n` palabras más frecuentes, ya ordenadas por construcción."""
        start = 0 if include_unk else 1
        return [(self.itos[i], self.counts[i]) for i in range(start, min(start + n, len(self)))]

    # -- estadísticas ----------------------------------------------------

    @property
    def unk_count(self) -> int:
        """Ocurrencias del corpus que quedaron fuera del vocabulario."""
        return self.counts[self.unk_index]

    @property
    def coverage(self) -> float:
        """Proporción de tokens del corpus cubiertos por palabras reales."""
        if not self.total_tokens:
            return 0.0
        return 1.0 - self.unk_count / self.total_tokens

    def frequencies(self) -> list[float]:
        """Frecuencia relativa de cada índice respecto del total de tokens."""
        if not self.total_tokens:
            return [0.0] * len(self)
        return [c / self.total_tokens for c in self.counts]

    def keep_probabilities(self, threshold: float = 1e-5) -> list[float]:
        """Probabilidad de conservar cada índice bajo subsampling (ver `subsample`)."""
        probabilities = subsample(
            dict(zip(self.itos, self.counts)), threshold, total=self.total_tokens
        )
        return [probabilities[word] for word in self.itos]

    # -- serialización ---------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "unk_token": self.unk_token,
            "min_count": self.min_count,
            "total_tokens": self.total_tokens,
            "vocab_size": len(self),
            "coverage": self.coverage,
            "itos": self.itos,
            "stoi": self.stoi,
            "counts": self.counts,
        }

    @classmethod
    def from_counts(
        cls,
        counts: Mapping[str, int],
        total_tokens: int,
        vocab_size: int,
        *,
        min_count: int = 1,
        unk_token: str = UNK_TOKEN,
    ) -> "Vocabulary":
        """Arma el vocabulario a partir de un conteo ya calculado.

        Separar el conteo (caro, un pase completo por el corpus) de la selección
        del top-N (barata) permite comparar varios `vocab_size` o `min_count`
        sobre el mismo conteo sin volver a leer los 3 GB.

        Args:
            counts: frecuencia de cada palabra en el corpus.
            total_tokens: total de tokens del corpus, incluidos los que quedarán
                fuera del vocabulario.
            vocab_size: tamaño total del vocabulario, `<UNK>` incluido.
            min_count: frecuencia mínima para ser candidato.
        """
        if vocab_size < 2:
            raise ValueError(f"vocab_size debe ser >= 2 (incluye <UNK>), no {vocab_size!r}")
        if min_count < 1:
            raise ValueError(f"min_count debe ser >= 1, no {min_count!r}")

        # Orden determinista: frecuencia descendente y, a igual frecuencia, alfabético.
        candidates = [(w, c) for w, c in counts.items() if c >= min_count and w != unk_token]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        kept = candidates[: vocab_size - 1]

        covered = sum(count for _, count in kept)
        return cls(
            itos=[unk_token] + [word for word, _ in kept],
            counts=[total_tokens - covered] + [count for _, count in kept],
            total_tokens=total_tokens,
            unk_token=unk_token,
            min_count=min_count,
        )

    @classmethod
    def from_dict(cls, data: Mapping) -> "Vocabulary":
        # `stoi` se reconstruye desde `itos`; en el JSON se guarda solo para que
        # el archivo sea legible/usable por fuera de este código.
        return cls(
            itos=list(data["itos"]),
            counts=list(data["counts"]),
            total_tokens=int(data["total_tokens"]),
            unk_token=data.get("unk_token", UNK_TOKEN),
            min_count=int(data.get("min_count", 1)),
        )

    def save(self, path: str | Path) -> Path:
        """Guarda el vocabulario como JSON (crea el directorio si hace falta)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def count_word_frequencies(
    path: str | Path,
    sample_fraction: float = 1.0,
    *,
    limit: int | None = None,
    seed: int = 42,
    progress: bool = False,
    max_types: int | None = None,
    **tokenizer_kwargs,
) -> tuple[Counter, int, int]:
    """Cuenta la frecuencia de cada palabra recorriendo el corpus por streaming.

    Args:
        max_types: cota superior aproximada de tipos distintos en memoria. Al
            superarla se descartan los tipos menos frecuentes vistos hasta ese
            momento (misma estrategia que gensim). Vuelve **aproximados** los
            conteos de la cola larga —una palabra podada y luego revista arranca
            de cero—, pero no afecta a las palabras frecuentes, que son las
            únicas que entran al vocabulario. Con `None` el conteo es exacto.
        Los demás argumentos se pasan a `src.corpus.stream_tokens`.

    Returns:
        `(counts, total_tokens, n_sentences)`. `total_tokens` y `n_sentences` son
        exactos aunque haya habido poda.
    """
    counts: Counter = Counter()
    total_tokens = 0
    n_sentences = 0
    prune_threshold = 1

    sentences = stream_tokens(
        path, sample_fraction, limit=limit, seed=seed, **tokenizer_kwargs
    )
    if progress:
        try:
            from tqdm.auto import tqdm

            sentences = tqdm(sentences, unit=" oraciones", unit_scale=True)
        except ImportError:
            pass

    for tokens in sentences:
        counts.update(tokens)
        total_tokens += len(tokens)
        n_sentences += 1
        if max_types is not None and len(counts) > max_types:
            for word in [w for w, c in counts.items() if c <= prune_threshold]:
                del counts[word]
            prune_threshold += 1

    return counts, total_tokens, n_sentences


def build_vocabulary(
    path: str | Path,
    vocab_size: int,
    min_count: int = 1,
    *,
    sample_fraction: float = 1.0,
    limit: int | None = None,
    seed: int = 42,
    progress: bool = False,
    max_types: int | None = None,
    unk_token: str = UNK_TOKEN,
    **tokenizer_kwargs,
) -> Vocabulary:
    """Construye el vocabulario: top-N palabras del corpus más `<UNK>`.

    Args:
        vocab_size: tamaño **total** del vocabulario, `<UNK>` incluido. Con
            `vocab_size=5000` se conservan las 4.999 palabras más frecuentes y el
            resto del corpus cae en `<UNK>`; así el tamaño coincide con el número
            de filas de la matriz de embeddings.
        min_count: frecuencia mínima para que una palabra sea candidata. Puede
            dejar el vocabulario por debajo de `vocab_size` si el corpus es chico.
        Los demás argumentos se pasan a `count_word_frequencies`.

    Returns:
        El `Vocabulary` construido.
    """
    counts, total_tokens, _ = count_word_frequencies(
        path,
        sample_fraction,
        limit=limit,
        seed=seed,
        progress=progress,
        max_types=max_types,
        **tokenizer_kwargs,
    )
    return Vocabulary.from_counts(
        counts,
        total_tokens,
        vocab_size,
        min_count=min_count,
        unk_token=unk_token,
    )


def build_vocabulary_from_config(
    config: Mapping,
    *,
    limit: int | None = None,
    progress: bool = True,
    max_types: int | None = None,
) -> Vocabulary:
    """Atajo que arma el vocabulario tomando todos los parámetros de un YAML.

    `limit` permite recortar el corpus para una prueba rápida sin tocar la
    configuración (útil desde los notebooks).
    """
    config = dict(config)
    return build_vocabulary(
        corpus_path(config),
        vocab_size=config["vocab_size"],
        min_count=config.get("min_count", 1),
        sample_fraction=config.get("corpus_sample_fraction", 1.0),
        limit=limit,
        seed=config.get("seed", 42),
        progress=progress,
        max_types=max_types,
        **config.get("tokenizer", {}),
    )


def subsample(
    word_counts: Mapping[str, int] | Sequence[tuple[str, int]],
    threshold: float = 1e-5,
    *,
    total: int | None = None,
) -> dict[str, float]:
    """Subsampling de palabras frecuentes (Mikolov et al., 2013).

    Cada ocurrencia de una palabra se conserva con probabilidad

        P_keep(w) = min(1, sqrt(t / f(w)))

    donde `f(w)` es la frecuencia relativa de `w` en el corpus y `t` el umbral.
    Las palabras muy frecuentes ("de", "la", "que") se descartan casi siempre,
    lo que acelera el entrenamiento y mejora los embeddings de las palabras poco
    frecuentes; las que están por debajo del umbral se conservan siempre.

    Args:
        word_counts: mapeo palabra→frecuencia (o iterable de pares).
        threshold: umbral `t`. Con `<= 0` el subsampling queda desactivado y
            todas las probabilidades valen 1.
        total: total de tokens del corpus. Por defecto se usa la suma de
            `word_counts`; conviene pasarlo explícitamente cuando los conteos
            son solo los del vocabulario y no los del corpus entero.

    Returns:
        Diccionario palabra → probabilidad de conservar cada ocurrencia.
    """
    counts = dict(word_counts)
    if threshold <= 0:
        return {word: 1.0 for word in counts}

    denominator = total if total is not None else sum(counts.values())
    if not denominator:
        return {word: 1.0 for word in counts}

    probabilities: dict[str, float] = {}
    for word, count in counts.items():
        frequency = count / denominator
        if frequency <= 0:
            probabilities[word] = 1.0
        else:
            probabilities[word] = min(1.0, math.sqrt(threshold / frequency))
    return probabilities


def _main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(description="Construye el vocabulario de una corrida.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--limit", type=int, default=None, help="usar solo las primeras N oraciones"
    )
    parser.add_argument(
        "--max-types",
        type=int,
        default=None,
        help="cota de tipos distintos en memoria (poda aproximada de la cola larga)",
    )
    parser.add_argument("--output", default=None, help="ruta del vocab.json de salida")
    parser.add_argument("--top", type=int, default=20, help="palabras a mostrar")
    args = parser.parse_args()

    config = load_config(args.config)

    # Una config con `vocab_from` toma prestado el vocabulario de otra corrida.
    # Reconstruirlo desde acá sobrescribiría el de la corrida dueña con datos de
    # un corpus distinto, que es justo lo contrario de lo que se busca.
    if reuses_vocabulary(config) and not args.output:
        raise SystemExit(
            f"'{config['name']}' reutiliza el vocabulario de "
            f"'{vocab_owner(config)}' (clave vocab_from), así que no se "
            f"reconstruye desde esta config.\n"
            f"  - Para regenerar ese vocabulario: usar la config de "
            f"'{vocab_owner(config)}'.\n"
            f"  - Para construir uno propio igualmente: pasar --output."
        )

    print(f"config: {config['name']} | vocab_size={config['vocab_size']} "
          f"min_count={config.get('min_count', 1)}")

    vocab = build_vocabulary_from_config(
        config, limit=args.limit, progress=True, max_types=args.max_types
    )

    destination = Path(args.output) if args.output else vocab_path(config, create=True)
    vocab.save(destination)

    print(vocab)
    print(f"tokens totales     : {vocab.total_tokens:,}")
    print(f"tokens en <UNK>    : {vocab.unk_count:,} ({1 - vocab.coverage:.2%})")
    print(f"cobertura          : {vocab.coverage:.2%}")
    print(f"guardado en        : {destination}")
    print(f"\ntop {args.top}:")
    for rank, (word, count) in enumerate(vocab.most_common(args.top), start=1):
        print(f"  {rank:3d}. {word:<15s} {count:>12,}")


if __name__ == "__main__":
    _main()
