"""Fase 6 — Evaluación de los embeddings entrenados.

Tres formas de mirar si los vectores capturan significado:

* **Vecinos cercanos** — ¿qué palabras quedaron cerca de una dada? Es la prueba
  cualitativa: si `lunes` no trae los otros días de la semana, algo está mal.
* **Analogías** — `mujer - hombre + rey ≈ reina`. Prueba que ciertas relaciones
  quedaron codificadas como *direcciones* consistentes en el espacio.
* **Precisión sobre un set de analogías** — la versión cuantitativa de lo
  anterior, para poder comparar configuraciones con un número.

Todo se hace sobre embeddings normalizados, así el producto punto entre dos
vectores **es** su similitud coseno.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from src import configure_console
from src.config import checkpoint_dir, load_config, vocab_path
from src.vocabulary import Vocabulary

__all__ = [
    "ANALOGIES",
    "load_embeddings",
    "normalize_rows",
    "similarity",
    "nearest_neighbors",
    "word_analogy",
    "evaluate_analogies",
]

#: Set de analogías en español para medir la calidad de forma comparable entre
#: configuraciones. Cada categoría es una lista de pares que comparten la misma
#: relación; las analogías se arman combinando pares dentro de una categoría.
#: Las palabras que no estén en el vocabulario se saltean y se reportan aparte,
#: así el mismo set sirve para vocabularios de 5.000 y de 20.000.
ANALOGIES: dict[str, list[tuple[str, str]]] = {
    "género": [
        ("hombre", "mujer"), ("rey", "reina"), ("niño", "niña"),
        ("padre", "madre"), ("hermano", "hermana"), ("actor", "actriz"),
        ("esposo", "esposa"), ("hijo", "hija"), ("señor", "señora"),
        ("presidente", "presidenta"), ("alumno", "alumna"), ("doctor", "doctora"),
    ],
    "plural": [
        ("año", "años"), ("día", "días"), ("casa", "casas"), ("país", "países"),
        ("ciudad", "ciudades"), ("hombre", "hombres"), ("empresa", "empresas"),
        ("libro", "libros"), ("mes", "meses"), ("grupo", "grupos"),
    ],
    "verbo: presente → pasado": [
        ("dice", "dijo"), ("hace", "hizo"), ("tiene", "tuvo"), ("va", "fue"),
        ("está", "estuvo"), ("puede", "pudo"), ("viene", "vino"), ("da", "dio"),
    ],
    "país → gentilicio": [
        ("argentina", "argentino"), ("chile", "chileno"), ("españa", "español"),
        ("italia", "italiano"), ("francia", "francés"), ("méxico", "mexicano"),
        ("brasil", "brasileño"), ("colombia", "colombiano"),
        ("uruguay", "uruguayo"), ("perú", "peruano"),
    ],
    "adjetivo → adverbio": [
        ("rápido", "rápidamente"), ("claro", "claramente"),
        ("general", "generalmente"), ("normal", "normalmente"),
        ("directo", "directamente"),
    ],
}


def normalize_rows(matrix: np.ndarray, *, epsilon: float = 1e-12) -> np.ndarray:
    """Lleva cada fila a norma 1.

    Con las filas normalizadas, el producto punto entre dos vectores es
    directamente su similitud coseno, que es lo que interesa: importa la
    *dirección* del vector, no su longitud. Las palabras frecuentes tienden a
    tener vectores más largos solo por haberse actualizado más veces, y eso no
    debería contar como parecido.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, epsilon)


def load_embeddings(
    config: Mapping,
    checkpoint: str = "best.pt",
    *,
    normalize: bool = True,
) -> tuple[np.ndarray, Vocabulary]:
    """Carga la matriz de embeddings de una corrida entrenada.

    Devuelve `input_embeddings`, que es la representación de cada palabra *como
    contexto*. La otra matriz del modelo (`output_embeddings`) se descarta: solo
    sirvió para puntuar durante el entrenamiento.

    Args:
        config: configuración de la corrida.
        checkpoint: nombre del archivo dentro de `checkpoints/{name}/`.
        normalize: normalizar las filas (recomendado, ver `normalize_rows`).

    Returns:
        `(embeddings, vocabulary)` con la matriz como `float32` de `(V, D)`.
    """
    import torch  # local: evaluar no debería exigir torch si ya se exportó

    path = checkpoint_dir(config) / checkpoint
    if not path.exists():
        raise FileNotFoundError(
            f"no existe {path}. ¿Se entrenó esta configuración?\n"
            f"  python -m src.train --config configs/{config['name']}.yaml"
        )

    data = torch.load(path, map_location="cpu", weights_only=False)
    weights = data["model_state"]["input_embeddings.weight"].numpy().astype(np.float32)
    vocab = Vocabulary.load(vocab_path(config))

    if len(vocab) != len(weights):
        raise ValueError(
            f"el checkpoint tiene {len(weights)} filas pero el vocabulario "
            f"{len(vocab)} palabras"
        )

    return (normalize_rows(weights) if normalize else weights), vocab


def similarity(
    first: str, second: str, embeddings: np.ndarray, vocab: Vocabulary
) -> float:
    """Similitud coseno entre dos palabras. 1 = idénticas, 0 = sin relación."""
    for word in (first, second):
        if word not in vocab:
            raise KeyError(f"{word!r} no está en el vocabulario")
    return float(embeddings[vocab.index(first)] @ embeddings[vocab.index(second)])


def nearest_neighbors(
    word: str | np.ndarray,
    embeddings: np.ndarray,
    vocab: Vocabulary,
    k: int = 10,
    *,
    exclude: Iterable[str] = (),
) -> list[tuple[str, float]]:
    """Las `k` palabras más parecidas, por similitud coseno.

    Args:
        word: palabra del vocabulario, o directamente un vector (para consultar
            el resultado de una operación aritmética, como en las analogías).
        embeddings: matriz normalizada.
        vocab: vocabulario.
        k: cuántos vecinos devolver.
        exclude: palabras a omitir del resultado, además de la consultada.

    Returns:
        Lista de `(palabra, similitud)` ordenada de mayor a menor.
    """
    excluded = {vocab.index(w) for w in exclude if w in vocab}
    if isinstance(word, str):
        if word not in vocab:
            raise KeyError(f"{word!r} no está en el vocabulario")
        vector = embeddings[vocab.index(word)]
        excluded.add(vocab.index(word))
    else:
        vector = np.asarray(word, dtype=np.float32)
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)

    scores = embeddings @ vector
    # `<UNK>` nunca es una respuesta útil: con drop_unknown=True su vector jamás
    # se entrenó y quedó en la inicialización aleatoria.
    excluded.add(vocab.unk_index)

    order = np.argsort(-scores)
    result: list[tuple[str, float]] = []
    for index in order:
        if index in excluded:
            continue
        result.append((vocab.word(int(index)), float(scores[index])))
        if len(result) >= k:
            break
    return result


def word_analogy(
    a: str,
    b: str,
    c: str,
    embeddings: np.ndarray,
    vocab: Vocabulary,
    k: int = 5,
) -> list[tuple[str, float]]:
    """Resuelve "`a` es a `b` como `c` es a ¿?".

    Ejemplo clásico: `word_analogy("hombre", "mujer", "rey")` debería devolver
    "reina". La cuenta es `b - a + c`: se toma la *dirección* que va de `a` a `b`
    (que en ese ejemplo representa "cambiar de masculino a femenino") y se la
    aplica a `c`.

    Las tres palabras de entrada se excluyen del resultado, porque suelen quedar
    entre las más cercanas al vector consultado y taparían la respuesta buscada.
    """
    for word in (a, b, c):
        if word not in vocab:
            raise KeyError(f"{word!r} no está en el vocabulario")

    vector = embeddings[vocab.index(b)] - embeddings[vocab.index(a)] + embeddings[vocab.index(c)]
    return nearest_neighbors(vector, embeddings, vocab, k, exclude=(a, b, c))


def evaluate_analogies(
    embeddings: np.ndarray,
    vocab: Vocabulary,
    analogies: Mapping[str, Sequence[tuple[str, str]]] | None = None,
    *,
    k: int = 1,
    detail: bool = False,
) -> dict:
    """Precisión sobre un set de analogías, por categoría y global.

    Para cada categoría se combinan todos los pares entre sí: con los pares
    `(hombre, mujer)` y `(rey, reina)` se plantea "hombre es a mujer como rey es
    a ¿?" y se cuenta acierto si "reina" aparece entre las `k` primeras.

    Las analogías con alguna palabra fuera del vocabulario se saltean y se
    reportan en `omitidas`, para que el mismo set sirva para comparar
    vocabularios de distinto tamaño.

    Args:
        detail: agregar la lista `items` con el resultado ítem por ítem, como
            `(categoría, a, b, c, esperada, acertó)`. Sirve para comparar dos
            modelos **de a pares** sobre las mismas analogías (ver
            `src.compare.mcnemar`), que es bastante más sensible que comparar
            dos porcentajes por separado.

    Returns:
        Diccionario con la precisión por categoría, la global, y los conteos.
    """
    analogies = analogies or ANALOGIES
    by_category: dict[str, dict] = {}
    items: list[tuple[str, str, str, str, str, bool]] = []
    total_correct = 0
    total_evaluated = 0
    total_skipped = 0

    for category, pairs in analogies.items():
        usable = [p for p in pairs if p[0] in vocab and p[1] in vocab]
        skipped = len(pairs) - len(usable)
        correct = 0
        evaluated = 0

        for i, (a, b) in enumerate(usable):
            for j, (c, expected) in enumerate(usable):
                if i == j:
                    continue
                predictions = word_analogy(a, b, c, embeddings, vocab, k=k)
                hit = expected in {word for word, _ in predictions}
                correct += hit
                evaluated += 1
                if detail:
                    items.append((category, a, b, c, expected, hit))

        by_category[category] = {
            "correct": correct,
            "evaluated": evaluated,
            "accuracy": correct / evaluated if evaluated else float("nan"),
            "pairs_used": len(usable),
            "pairs_skipped": skipped,
        }
        total_correct += correct
        total_evaluated += evaluated
        total_skipped += skipped

    resultado = {
        "k": k,
        "categories": by_category,
        "correct": total_correct,
        "evaluated": total_evaluated,
        "accuracy": total_correct / total_evaluated if total_evaluated else float("nan"),
        "pairs_skipped": total_skipped,
    }
    if detail:
        resultado["items"] = items
    return resultado


def _main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(description="Evalúa embeddings entrenados.")
    parser.add_argument("--config", default="configs/piloto_5k_50_2.yaml")
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--k", type=int, default=8, help="vecinos a mostrar")
    parser.add_argument(
        "--words",
        nargs="*",
        default=["lunes", "febrero", "tres", "rojo", "presidente", "argentina"],
    )
    args = parser.parse_args()

    config = load_config(args.config)
    embeddings, vocab = load_embeddings(config, args.checkpoint)
    print(f"config: {config['name']} | embeddings {embeddings.shape}\n")

    print("VECINOS CERCANOS")
    for word in args.words:
        if word not in vocab:
            print(f"  {word:<14} (fuera del vocabulario)")
            continue
        vecinos = nearest_neighbors(word, embeddings, vocab, args.k)
        print(f"  {word:<14} -> " + ", ".join(f"{w} ({s:.2f})" for w, s in vecinos))

    print("\nANALOGÍAS")
    for a, b, c in [("hombre", "mujer", "rey"), ("madrid", "españa", "roma"),
                    ("dice", "dijo", "hace")]:
        if not all(w in vocab for w in (a, b, c)):
            print(f"  {a} : {b} :: {c} : ?   (alguna palabra fuera del vocabulario)")
            continue
        result = word_analogy(a, b, c, embeddings, vocab, k=3)
        print(f"  {a} : {b} :: {c} : " + ", ".join(f"{w} ({s:.2f})" for w, s in result))

    print("\nPRECISIÓN SOBRE EL SET DE ANALOGÍAS")
    report = evaluate_analogies(embeddings, vocab, k=1)
    for category, data in report["categories"].items():
        print(f"  {category:<26} {data['accuracy']:>6.1%}  "
              f"({data['correct']}/{data['evaluated']})")
    print(f"  {'GLOBAL':<26} {report['accuracy']:>6.1%}  "
          f"({report['correct']}/{report['evaluated']})")


if __name__ == "__main__":
    _main()
