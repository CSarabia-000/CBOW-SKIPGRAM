"""Fase 6 — Exportación de los embeddings al formato Word2Vec estándar.

El formato lo definió el word2vec original y lo entienden gensim, spaCy y casi
cualquier herramienta de NLP. Exportarlo es lo que hace que los embeddings
sirvan **fuera** de este proyecto, sin depender de PyTorch ni de este código.

Formato de texto (`.txt`):

    5000 50
    de 0.123456 -0.234567 ... 0.345678
    la -0.111111 0.222222 ... -0.333333

Primera línea: cantidad de palabras y dimensión. Después una línea por palabra,
con la palabra, un espacio, y los valores separados por espacios.

Formato binario (`.bin`): igual encabezado, pero cada vector se guarda como
bytes `float32` en vez de texto. Ocupa la mitad y carga mucho más rápido, a
costa de no ser legible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src import configure_console
from src.config import PROJECT_ROOT, load_config
from src.evaluate import load_embeddings
from src.vocabulary import Vocabulary

__all__ = [
    "embeddings_path",
    "export_word2vec_format",
    "export_from_config",
]


def embeddings_path(config: dict, *, binary: bool = False, create: bool = False) -> Path:
    """Ruta de salida de una corrida: `embeddings/{name}.txt` (o `.bin`)."""
    directory = PROJECT_ROOT / "embeddings"
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{config['name']}.{'bin' if binary else 'txt'}"


def export_word2vec_format(
    embeddings: np.ndarray,
    vocab: Vocabulary,
    output_path: str | Path,
    *,
    binary: bool = False,
    include_unk: bool = False,
    precision: int = 6,
) -> Path:
    """Exporta la matriz de embeddings al formato Word2Vec.

    Args:
        embeddings: matriz `(V, D)`.
        vocab: vocabulario; su orden define el de las filas.
        output_path: archivo de salida.
        binary: usar el formato binario en vez del de texto.
        include_unk: incluir `<UNK>`. Por defecto **no**, y con razón: con
            `drop_unknown=True` ese token nunca aparece como contexto ni como
            target, así que su vector jamás recibió un gradiente y sigue siendo
            el ruido de la inicialización. Exportarlo sería publicar un vector
            sin significado.
        precision: decimales en el formato de texto.

    Returns:
        La ruta escrita.
    """
    if len(vocab) != len(embeddings):
        raise ValueError(
            f"el vocabulario tiene {len(vocab)} palabras y la matriz "
            f"{len(embeddings)} filas"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start = 0 if include_unk else 1
    words = vocab.itos[start:]
    vectors = np.asarray(embeddings[start:], dtype=np.float32)
    header = f"{len(words)} {vectors.shape[1]}\n"

    if binary:
        with output_path.open("wb") as handle:
            handle.write(header.encode("utf-8"))
            for word, vector in zip(words, vectors):
                # El formato del word2vec original: la palabra, un espacio, y
                # los bytes crudos del vector, sin separador al final.
                handle.write(word.encode("utf-8"))
                handle.write(b" ")
                handle.write(vector.tobytes())
    else:
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(header)
            for word, vector in zip(words, vectors):
                valores = " ".join(f"{v:.{precision}f}" for v in vector)
                handle.write(f"{word} {valores}\n")

    return output_path


def export_from_config(
    config: dict,
    *,
    checkpoint: str = "best.pt",
    binary: bool = False,
    normalize: bool = False,
    include_unk: bool = False,
) -> Path:
    """Exporta los embeddings de una corrida entrenada.

    Args:
        normalize: exportar los vectores normalizados a norma 1. Por defecto
            **no**: conviene guardar los vectores tal como quedaron y dejar que
            quien los use decida. gensim normaliza por su cuenta al calcular
            similitudes, y la norma original lleva información que se perdería.

            Medido en la corrida piloto, esa información no es la que uno
            esperaría: las palabras *más* frecuentes terminan con vectores más
            **cortos** (`de`, la primera, tiene norma 1,41; `italia`, la número
            mil, tiene 2,83). Aparecen en contextos tan variados que sus
            actualizaciones se cancelan entre sí, mientras que una palabra
            temática se ve siempre en el mismo ambiente y su vector se estira en
            una dirección clara.
    """
    embeddings, vocab = load_embeddings(config, checkpoint, normalize=normalize)
    destination = embeddings_path(config, binary=binary, create=True)
    return export_word2vec_format(
        embeddings, vocab, destination, binary=binary, include_unk=include_unk
    )


def _main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(
        description="Exporta embeddings al formato Word2Vec."
    )
    parser.add_argument("--config", default="configs/piloto_5k_50_2.yaml")
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--binary", action="store_true", help="formato .bin")
    parser.add_argument(
        "--normalize", action="store_true", help="exportar vectores de norma 1"
    )
    parser.add_argument("--include-unk", action="store_true")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="recargar el archivo con gensim para validar el formato",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    destination = export_from_config(
        config,
        checkpoint=args.checkpoint,
        binary=args.binary,
        normalize=args.normalize,
        include_unk=args.include_unk,
    )
    size = destination.stat().st_size
    print(f"exportado: {destination}")
    print(f"tamaño   : {size / 1024**2:.2f} MB")

    if args.verify:
        from gensim.models import KeyedVectors

        vectors = KeyedVectors.load_word2vec_format(destination, binary=args.binary)
        print(f"gensim    : {len(vectors.index_to_key):,} palabras, "
              f"dimensión {vectors.vector_size}")
        print(f"vecinos de 'lunes': "
              f"{[w for w, _ in vectors.most_similar('lunes', topn=5)]}")


if __name__ == "__main__":
    _main()
