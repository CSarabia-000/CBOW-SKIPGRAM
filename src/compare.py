"""Fase 7 — Comparación de configuraciones entrenadas.

Reúne en una sola tabla lo que dejó cada corrida (`history.json`, checkpoints,
embeddings exportados) y le agrega las métricas de calidad, para poder decidir
qué conviene escalar. No entrena nada: solo lee artefactos ya producidos.

Tres cuidados al comparar, que este módulo hace explícitos porque son fáciles de
pasar por alto:

1. **La loss no cruza tamaños de contexto.** Predecir un target desde hasta 10
   palabras es una tarea más fácil que desde 4, así que una corrida con
   `context_size=5` tendrá menor loss que una con `context_size=2` aunque no
   haya aprendido nada mejor. `comparable_loss` marca qué filas se pueden
   comparar entre sí. La métrica que sí cruza todas las configuraciones es la
   precisión en analogías.

2. **La precisión en analogías tiene ruido.** Son unos 286 ítems: al 41% de
   precisión, el intervalo de confianza del 95% mide unos ±6 puntos. Por eso
   `run_summary` devuelve el intervalo de Wilson junto al porcentaje, y por eso
   existe `mcnemar`: comparar dos modelos **sobre los mismos ítems** es mucho
   más sensible que comparar dos porcentajes con sus intervalos.

3. **Un modelo entrenado con `<UNK>` no se evalúa con la loss de su propia
   validación.** Predecir el token más frecuente del corpus es trivial y le baja
   la loss sin que eso sea aprender mejor. `cross_validation_loss` puntúa
   cualquier modelo sobre el stream de validación de *otra* configuración, para
   que los dos números midan la misma tarea.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src import configure_console
from src.config import checkpoint_dir, load_config
from src.evaluate import evaluate_analogies, load_embeddings, nearest_neighbors
from src.export import embeddings_path
from src.train import load_history
from src.vocabulary import Vocabulary

__all__ = [
    "GRID",
    "wilson_interval",
    "mcnemar",
    "run_summary",
    "collect",
    "cross_validation_loss",
]

#: Las cuatro celdas del factorial 2x2 de la Fase 7 (vocabulario y corpus fijos).
GRID = [
    "piloto_5k_50_2",
    "piloto_5k_100_2",
    "piloto_5k_50_5",
    "piloto_5k_100_5",
]


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confianza de Wilson para una proporción.

    Se usa este y no el `p ± z·sqrt(p(1-p)/n)` de manual porque el de manual se
    porta mal con proporciones cerca de 0 o de 1 y con muestras chicas — que es
    justo el caso de las categorías de 20 ítems.
    """
    if total == 0:
        return (float("nan"), float("nan"))
    p = correct / total
    denom = 1 + z**2 / total
    centro = (p + z**2 / (2 * total)) / denom
    margen = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, centro - margen), min(1.0, centro + margen))


def mcnemar(
    hits_a: Sequence[bool], hits_b: Sequence[bool]
) -> dict:
    """Prueba de McNemar entre dos modelos evaluados sobre los MISMOS ítems.

    Comparar `41,3%` contra `44,1%` con intervalos separados desperdicia
    información: los dos modelos respondieron exactamente las mismas analogías,
    así que lo que importa no es cuántas acertó cada uno sino **en cuántas
    difieren**. McNemar mira solo los desacuerdos: `solo_a` (acertó A y falló B)
    y `solo_b` (al revés). Si los modelos fueran equivalentes, cada desacuerdo
    sería una moneda al aire.

    El p-valor es el binomial exacto de dos colas con p=0,5, sin aproximación
    ji-cuadrado, que con pocos desacuerdos es poco confiable.

    Returns:
        `solo_a`, `solo_b`, `ambos`, `ninguno`, `n_desacuerdos` y `p_value`.
    """
    if len(hits_a) != len(hits_b):
        raise ValueError(
            f"los dos modelos deben evaluarse sobre los mismos ítems: "
            f"{len(hits_a)} != {len(hits_b)}"
        )

    solo_a = sum(1 for a, b in zip(hits_a, hits_b) if a and not b)
    solo_b = sum(1 for a, b in zip(hits_a, hits_b) if b and not a)
    ambos = sum(1 for a, b in zip(hits_a, hits_b) if a and b)
    ninguno = sum(1 for a, b in zip(hits_a, hits_b) if not a and not b)

    n = solo_a + solo_b
    if n == 0:
        p_value = 1.0
    else:
        cola = sum(math.comb(n, i) for i in range(min(solo_a, solo_b) + 1))
        p_value = min(1.0, 2 * cola / 2**n)

    return {
        "solo_a": solo_a,
        "solo_b": solo_b,
        "ambos": ambos,
        "ninguno": ninguno,
        "n_desacuerdos": n,
        "p_value": p_value,
    }


def _checkpoint_size(config: Mapping) -> int:
    path = checkpoint_dir(config) / "best.pt"
    return path.stat().st_size if path.exists() else 0


def run_summary(
    config: Mapping | str | Path,
    *,
    checkpoint: str = "best.pt",
    k: int = 1,
    detail: bool = True,
) -> dict:
    """Todo lo que hay que saber de una corrida entrenada, en un diccionario.

    Junta los hiperparámetros que la definen, lo que quedó registrado en su
    `history.json` (tiempos, curvas) y la calidad medida sobre sus embeddings.
    """
    if not isinstance(config, Mapping):
        config = load_config(config)

    history = load_history(config)
    if not history:
        raise FileNotFoundError(
            f"la corrida {config['name']!r} no tiene history.json: ¿se entrenó?"
        )

    embeddings, vocab = load_embeddings(config, checkpoint)
    analogias = evaluate_analogies(embeddings, vocab, k=k, detail=detail)
    lo, hi = wilson_interval(analogias["correct"], analogias["evaluated"])

    val = [r["val_loss"] for r in history if r.get("val_loss") is not None]
    txt = embeddings_path(config)
    binario = embeddings_path(config, binary=True)

    return {
        "name": config["name"],
        "embedding_dim": config["embedding_dim"],
        "context_size": config["context_size"],
        "drop_unknown": config.get("drop_unknown", True),
        # La loss solo es comparable entre corridas que resuelven la misma
        # tarea: mismo contexto y mismo tratamiento de los OOV.
        "comparable_loss": (config["context_size"], config.get("drop_unknown", True)),
        "epochs": len(history),
        "params": len(vocab) * config["embedding_dim"] * 2,
        "train_loss": history[-1]["train_loss"],
        "val_loss": val[-1] if val else None,
        "best_val_loss": min(val) if val else None,
        "pairs_per_epoch": history[-1]["pairs"],
        "seconds_total": sum(r["seconds"] for r in history),
        "seconds_per_epoch": sum(r["seconds"] for r in history) / len(history),
        "pairs_per_second": history[-1]["pairs"] / history[-1]["seconds"],
        "accuracy": analogias["accuracy"],
        "accuracy_correct": analogias["correct"],
        "accuracy_evaluated": analogias["evaluated"],
        "accuracy_ci": (lo, hi),
        "categories": analogias["categories"],
        "items": analogias.get("items"),
        "checkpoint_bytes": _checkpoint_size(config),
        "embeddings_txt_bytes": txt.stat().st_size if txt.exists() else 0,
        "embeddings_bin_bytes": binario.stat().st_size if binario.exists() else 0,
        "history": history,
    }


def collect(
    names: Iterable[str] | None = None, *, k: int = 1, detail: bool = True
) -> list[dict]:
    """`run_summary` para varias corridas, en el orden pedido."""
    names = list(names) if names is not None else GRID
    return [
        run_summary(f"configs/{name}.yaml", k=k, detail=detail) for name in names
    ]


def cross_validation_loss(
    model_config: Mapping | str | Path,
    data_config: Mapping | str | Path,
    *,
    checkpoint: str = "best.pt",
    device: str | None = None,
    max_batches: int | None = None,
    seed: int = 42,
) -> float:
    """Loss de un modelo sobre el conjunto de validación de *otra* configuración.

    Existe por la ablación de `<UNK>`: los dos modelos comparten dimensión y
    contexto, pero uno entrenó con los OOV convertidos en `<UNK>` y el otro con
    los OOV eliminados, así que sus validaciones **no** miden lo mismo. Puntuando
    ambos sobre el mismo stream (el de `data_config`) los dos números vuelven a
    ser la misma pregunta.

    `model_config` y `data_config` tienen que coincidir en `embedding_dim` y en
    `context_size`, o los tensores no encajan.
    """
    import torch
    from torch.utils.data import DataLoader

    from src.config import vocab_path
    from src.dataset import CBOWIterableDataset
    from src.model import build_model_from_config
    from src.train import _resolve_device, evaluate_loss, load_checkpoint

    if not isinstance(model_config, Mapping):
        model_config = load_config(model_config)
    if not isinstance(data_config, Mapping):
        data_config = load_config(data_config)

    for clave in ("embedding_dim", "context_size"):
        if model_config[clave] != data_config[clave]:
            raise ValueError(
                f"no se pueden cruzar configuraciones con distinto {clave}: "
                f"{model_config[clave]} vs {data_config[clave]}"
            )

    device_obj = _resolve_device(device)
    vocab = Vocabulary.load(vocab_path(model_config))
    model, _ = build_model_from_config(model_config, vocab)
    load_checkpoint(
        checkpoint_dir(model_config) / checkpoint, model, map_location=device_obj
    )
    model.to(device_obj)

    n_validation = int(data_config.get("validation_sentences", 0))
    dataset = CBOWIterableDataset.from_config(data_config, vocab, limit=n_validation)
    loader = DataLoader(
        dataset,
        batch_size=data_config["batch_size"],
        num_workers=0,
        pin_memory=device_obj.type == "cuda",
    )

    return evaluate_loss(
        model,
        vocab,
        loader,
        device_obj,
        data_config["negative_samples"],
        seed=seed,
        max_batches=max_batches,
    )


def _format_table(rows: Sequence[Mapping]) -> str:
    cabecera = (
        f"{'corrida':<22} {'dim':>4} {'ctx':>4} {'unk':>4} "
        f"{'val loss':>9} {'analogías':>10} {'IC 95%':>16} "
        f"{'min/época':>10} {'MB':>6}"
    )
    lineas = [cabecera, "-" * len(cabecera)]
    for r in rows:
        lo, hi = r["accuracy_ci"]
        val = "-" if r["val_loss"] is None else f"{r['val_loss']:.4f}"
        lineas.append(
            f"{r['name']:<22} {r['embedding_dim']:>4} {r['context_size']:>4} "
            f"{'sí' if not r['drop_unknown'] else 'no':>4} "
            f"{val:>9} "
            f"{r['accuracy']:>9.1%} "
            f"{f'[{lo:.1%}, {hi:.1%}]':>16} "
            f"{r['seconds_per_epoch'] / 60:>10.1f} "
            f"{r['checkpoint_bytes'] / 1024**2:>6.1f}"
        )
    return "\n".join(lineas)


def _main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(
        description="Compara varias configuraciones ya entrenadas."
    )
    parser.add_argument("--configs", nargs="*", default=GRID)
    parser.add_argument("--k", type=int, default=1, help="k de la analogía")
    parser.add_argument("--json", default=None, help="volcar el resumen a un .json")
    parser.add_argument(
        "--words",
        nargs="*",
        default=["febrero", "tres", "presidente", "mujer", "ciudades", "dijo"],
        help="palabras cuyos vecinos se muestran lado a lado",
    )
    args = parser.parse_args()

    filas = collect(args.configs, k=args.k)
    print(_format_table(filas))

    print("\nPRECISIÓN POR CATEGORÍA")
    categorias = list(filas[0]["categories"])
    ancho = max(len(c) for c in categorias)
    print(f"{'categoría':<{ancho}} " + " ".join(f"{r['name'][-8:]:>12}" for r in filas))
    for categoria in categorias:
        celdas = []
        for fila in filas:
            dato = fila["categories"][categoria]
            celdas.append(f"{dato['accuracy']:>7.1%} {dato['correct']:>2}/{dato['evaluated']:<2}")
        print(f"{categoria:<{ancho}} " + " ".join(f"{c:>12}" for c in celdas))

    if len(filas) >= 2 and all(f["items"] for f in filas):
        print("\nMcNEMAR contra la línea de base "
              f"({filas[0]['name']}), sobre los mismos ítems")
        base = [hit for *_, hit in filas[0]["items"]]
        for fila in filas[1:]:
            otro = [hit for *_, hit in fila["items"]]
            prueba = mcnemar(base, otro)
            marca = "*" if prueba["p_value"] < 0.05 else " "
            print(
                f"  {fila['name']:<22} base sola {prueba['solo_a']:>3} | "
                f"otra sola {prueba['solo_b']:>3} | p = {prueba['p_value']:.4f} {marca}"
            )

    print("\nVECINOS LADO A LADO")
    cargados = [
        (fila["name"], *load_embeddings(load_config(f"configs/{fila['name']}.yaml")))
        for fila in filas
    ]
    for palabra in args.words:
        print(f"\n  {palabra}")
        for nombre, embeddings, vocab in cargados:
            if palabra not in vocab:
                print(f"    {nombre:<22} (fuera del vocabulario)")
                continue
            vecinos = nearest_neighbors(palabra, embeddings, vocab, k=6)
            print(f"    {nombre:<22} " + ", ".join(w for w, _ in vecinos))

    if args.json:
        destino = Path(args.json)
        destino.parent.mkdir(parents=True, exist_ok=True)
        limpio = [
            {
                clave: valor
                for clave, valor in fila.items()
                if clave not in {"items", "history"}
            }
            for fila in filas
        ]
        destino.write_text(
            json.dumps(limpio, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nresumen escrito en {destino}")


if __name__ == "__main__":
    _main()
