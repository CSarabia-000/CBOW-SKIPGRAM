"""Fase 5 — Loop de entrenamiento, checkpoints y registro de la loss.

Arma el pipeline completo a partir de un YAML: vocabulario → dataset →
DataLoader → modelo → optimizador → épocas.

Sirve igual para CBOW y para skip-gram: el dataset y el modelo salen de sus
respectivas fábricas, que leen `arch` de la configuración, y el batch tiene la
misma forma `(entrada, máscara, target)` en los dos casos. Este módulo no
menciona ninguna arquitectura en particular a propósito; agregar una tercera no
debería obligarlo a cambiar.

Se puede ejecutar desde un notebook (`train("configs/base.yaml")`) o desde la
línea de comandos (`python -m src.train --config configs/base.yaml`).

Artefactos que deja en `checkpoints/{name}/`:

* `epoch_XX.pt` — un checkpoint por época (modelo + optimizador + historial).
* `last.pt` — la última época; es desde donde retoma `--resume`.
* `best.pt` — la de menor loss de validación (o de entrenamiento si no hay).
* `history.json` — solo las curvas, liviano, para comparar corridas sin cargar
  ningún modelo.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src import configure_console
from src.config import (
    checkpoint_dir,
    history_path,
    load_config,
    vocab_path,
)
from src.config import arch as config_arch
from src.dataset import build_dataset_from_config
from src.model import NegativeSampler, Word2VecModel, build_model_from_config
from src.vocabulary import Vocabulary

__all__ = [
    "EpochRecord",
    "evaluate_loss",
    "save_checkpoint",
    "load_checkpoint",
    "load_history",
    "train",
]


@dataclass
class EpochRecord:
    """Lo que se registra de cada época."""

    epoch: int
    train_loss: float
    val_loss: float | None
    steps: int
    pairs: int
    seconds: float
    learning_rate: float

    def to_dict(self) -> dict:
        return asdict(self)

    def format(self) -> str:
        val = "     -" if self.val_loss is None else f"{self.val_loss:6.4f}"
        return (
            f"época {self.epoch:>3} | train {self.train_loss:6.4f} | val {val} | "
            f"{self.pairs:>10,} pares | {self.seconds:>6.1f}s | "
            f"{self.pairs / self.seconds:>9,.0f} pares/s | lr {self.learning_rate:.2e}"
        )


def _resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _learning_rate(base: float, epoch: int, epochs: int, floor: float = 0.05) -> float:
    """Decaimiento lineal del learning rate a lo largo del entrenamiento.

    Es lo que hace el word2vec original: pasos grandes al principio, para moverse
    rápido, y cada vez más chicos al final, para asentarse en un mínimo en vez de
    seguir rebotando. `floor` evita que llegue exactamente a cero en la última
    época, donde el modelo dejaría de aprender del todo.
    """
    if epochs <= 1:
        return base
    return base * max(floor, 1.0 - epoch / epochs)


@torch.no_grad()
def evaluate_loss(
    model: Word2VecModel,
    vocab: Vocabulary,
    loader: DataLoader,
    device: torch.device,
    negative_samples: int,
    *,
    seed: int = 42,
    max_batches: int | None = None,
) -> float:
    """Loss media sobre un conjunto de validación.

    Para que la curva sea comparable entre épocas, la evaluación tiene que ser
    **determinista**: se usa un muestreador de negativos nuevo con semilla fija
    en cada llamada, así los negativos son siempre los mismos. El dataset de
    validación tampoco recibe `set_epoch`, de modo que sus pares no cambian.
    """
    model.eval()
    sampler = NegativeSampler(vocab.counts, seed=seed).to(device)

    total = 0.0
    batches = 0
    for context, mask, target in loader:
        context = context.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        negatives = sampler.sample(len(target), negative_samples)
        total += model.negative_sampling_loss(context, mask, target, negatives).item()
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break

    model.train()
    return total / batches if batches else float("nan")


def save_checkpoint(
    path: Path,
    model: Word2VecModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[EpochRecord],
    config: dict,
) -> Path:
    """Guarda modelo, optimizador e historial en un único archivo.

    El estado del optimizador se guarda para poder **retomar** exactamente donde
    se cortó: Adam mantiene medias móviles de los gradientes, y reanudar sin ellas
    equivale a reiniciar el optimizador a mitad de camino.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": [r.to_dict() for r in history],
            "config": config,
            "vocab_size": model.vocab_size,
            "embedding_dim": model.embedding_dim,
            "arch": model.ARCH,
        },
        path,
    )
    return path


def load_checkpoint(
    path: str | Path,
    model: Word2VecModel | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Carga un checkpoint y, si se pasan, restaura modelo y optimizador."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None:
        model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


def load_history(config: dict) -> list[dict]:
    """Lee `history.json` de una corrida. Lista vacía si todavía no entrenó."""
    path = history_path(config)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["epochs"]


def train(
    config_path: str | Path,
    *,
    device: str | None = None,
    num_workers: int = 0,
    resume: bool = False,
    max_steps_per_epoch: int | None = None,
    progress: bool = True,
) -> dict:
    """Entrena una configuración completa y devuelve su historial.

    Args:
        config_path: ruta al YAML de la corrida.
        device: `"cuda"`, `"cpu"` o `None` para detectar automáticamente.
        num_workers: procesos del `DataLoader`. El corpus se reparte por oración
            (ver `src.dataset`), así que la tokenización sí se paraleliza.
        resume: retomar desde `last.pt` si existe.
        max_steps_per_epoch: cortar cada época a N batches; para pruebas rápidas.
        progress: barra `tqdm` por época si está instalada.

    Returns:
        Diccionario con la configuración, el historial por época y las rutas de
        los artefactos generados.
    """
    config = load_config(config_path)
    torch.manual_seed(config.get("seed", 42))

    vocab = Vocabulary.load(vocab_path(config))
    device_obj = _resolve_device(device)
    model, sampler = build_model_from_config(config, vocab)
    model.to(device_obj)
    sampler.to(device_obj)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Las primeras `validation_sentences` oraciones se reservan para validar y
    # se saltean al entrenar, así ningún par aparece en los dos conjuntos.
    n_validation = int(config.get("validation_sentences", 0))
    train_dataset = build_dataset_from_config(
        config, vocab, skip_sentences=n_validation
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=num_workers,
        pin_memory=device_obj.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    validation_loader = None
    if n_validation > 0:
        validation_dataset = build_dataset_from_config(
            config, vocab, limit=n_validation
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=config["batch_size"],
            num_workers=0,
            pin_memory=device_obj.type == "cuda",
        )

    directory = checkpoint_dir(config, create=True)
    history: list[EpochRecord] = []
    first_epoch = 0

    if resume:
        last = directory / "last.pt"
        if last.exists():
            checkpoint = load_checkpoint(last, model, optimizer, map_location=device_obj)
            history = [EpochRecord(**r) for r in checkpoint["history"]]
            first_epoch = checkpoint["epoch"] + 1
            print(f"retomando desde la época {first_epoch} ({last.name})")
        else:
            print(f"no hay checkpoint en {last}, se entrena desde cero")

    epochs = config["epochs"]
    k = config["negative_samples"]
    seed = config.get("seed", 42)
    best = min((r.val_loss or r.train_loss for r in history), default=float("inf"))

    print(f"config     : {config['name']} | arch: {config_arch(config)}")
    print(f"dispositivo: {device_obj} | workers: {num_workers}")
    print(f"modelo     : {model}")
    print(f"validación : {n_validation:,} oraciones reservadas")
    print(f"epocas     : {first_epoch} a {epochs - 1}")
    print()

    try:
        for epoch in range(first_epoch, epochs):
            learning_rate = _learning_rate(config["learning_rate"], epoch, epochs)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            train_dataset.set_epoch(epoch)
            model.train()

            batches = train_loader
            if progress:
                try:
                    from tqdm.auto import tqdm

                    batches = tqdm(
                        train_loader, desc=f"época {epoch}", unit=" batch", leave=False
                    )
                except ImportError:
                    pass

            started = time.time()
            running = 0.0
            steps = 0
            pairs = 0

            for context, mask, target in batches:
                context = context.to(device_obj, non_blocking=True)
                mask = mask.to(device_obj, non_blocking=True)
                target = target.to(device_obj, non_blocking=True)

                negatives = sampler.sample(len(target), k)

                optimizer.zero_grad(set_to_none=True)
                loss = model.negative_sampling_loss(context, mask, target, negatives)
                loss.backward()
                optimizer.step()

                running += loss.item()
                steps += 1
                pairs += len(target)
                if max_steps_per_epoch is not None and steps >= max_steps_per_epoch:
                    break

            elapsed = time.time() - started
            train_loss = running / steps if steps else float("nan")

            val_loss = None
            if validation_loader is not None:
                val_loss = evaluate_loss(
                    model, vocab, validation_loader, device_obj, k, seed=seed
                )

            record = EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                steps=steps,
                pairs=pairs,
                seconds=elapsed,
                learning_rate=learning_rate,
            )
            history.append(record)
            print(record.format())

            save_checkpoint(
                directory / f"epoch_{epoch:02d}.pt", model, optimizer, epoch, history, config
            )
            save_checkpoint(directory / "last.pt", model, optimizer, epoch, history, config)

            score = val_loss if val_loss is not None else train_loss
            if score < best:
                best = score
                save_checkpoint(
                    directory / "best.pt", model, optimizer, epoch, history, config
                )

            _write_history(config, history, best)

    except KeyboardInterrupt:
        # Interrumpir a mano no debería costar el entrenamiento hecho hasta acá.
        print("\ninterrumpido: se conserva el último checkpoint guardado")

    print()
    print(f"artefactos en: {directory}")
    return {
        "config": config,
        "epochs": [r.to_dict() for r in history],
        "best": best,
        "checkpoint_dir": str(directory),
    }


def _write_history(config: dict, history: list[EpochRecord], best: float) -> None:
    history_path(config, create=True).write_text(
        json.dumps(
            {
                "name": config["name"],
                "config": config,
                "best": best,
                "epochs": [r.to_dict() for r in history],
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )


def _main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(
        description="Entrena la configuración indicada (CBOW o skip-gram)."
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--device", default=None, help="cuda, cpu o auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="retomar desde last.pt")
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=None,
        help="cortar cada época a N batches (para pruebas)",
    )
    args = parser.parse_args()

    train(
        args.config,
        device=args.device,
        num_workers=args.num_workers,
        resume=args.resume,
        max_steps_per_epoch=args.max_steps_per_epoch,
    )


if __name__ == "__main__":
    _main()
