"""Carga de configuraciones YAML y rutas derivadas de cada corrida.

Cada configuración (`configs/*.yaml`) define una corrida completa. Los módulos
reciben la ruta al YAML y de ahí sacan tanto los hiperparámetros como las rutas
de sus artefactos, para que escalar a otra combinación de la grilla sea solo
agregar un archivo nuevo en `configs/`.

La llave `arch` elige la arquitectura (`"cbow"` o `"skipgram"`). Es el único
lugar donde se decide: `src.dataset` y `src.model` despachan a partir de ella y
el resto del pipeline (corpus, vocabulario, entrenamiento, evaluación,
exportación) es idéntico para las dos. Las configuraciones escritas antes de que
existiera skip-gram no la declaran, así que el valor por defecto es `"cbow"`.
"""

from __future__ import annotations

from pathlib import Path

import yaml

__all__ = [
    "PROJECT_ROOT",
    "ARCHS",
    "load_config",
    "arch",
    "corpus_path",
    "processed_dir",
    "vocab_owner",
    "reuses_vocabulary",
    "vocab_path",
    "checkpoint_dir",
    "history_path",
]

#: Raíz del proyecto (el directorio que contiene `src/`).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Arquitecturas soportadas. `"cbow"` predice el target desde el promedio de su
#: contexto; `"skipgram"` predice cada palabra del contexto desde el target.
ARCHS = ("cbow", "skipgram")


def load_config(path: str | Path) -> dict:
    """Lee un YAML de configuración y devuelve su contenido como diccionario.

    Las rutas relativas dentro del YAML se interpretan siempre respecto de la
    raíz del proyecto, no del directorio de trabajo, para que el mismo archivo
    funcione desde un notebook en `notebooks_CBOW/` o `notebooks_SKIPGRAM/` o desde la línea de comandos.
    """
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{path} no contiene un diccionario de configuración")
    config.setdefault("name", path.stem)
    config.setdefault("tokenizer", {})
    config.setdefault("arch", "cbow")
    if config["arch"] not in ARCHS:
        raise ValueError(
            f"{path}: arch={config['arch']!r} no es una arquitectura conocida "
            f"(esperaba una de {ARCHS})"
        )
    return config


def arch(config: dict) -> str:
    """Arquitectura de la corrida: `"cbow"` o `"skipgram"`.

    Se lee con `.get()` y no con `config["arch"]` porque los diccionarios de
    configuración guardados dentro de checkpoints viejos —entrenados antes de
    que existiera skip-gram— no tienen la llave.
    """
    return config.get("arch", "cbow")


def corpus_path(config: dict) -> Path:
    """Ruta absoluta al corpus declarado en la configuración."""
    return PROJECT_ROOT / config["corpus_path"]


def processed_dir(config: dict, *, create: bool = False) -> Path:
    """Directorio de artefactos intermedios de la corrida: `data/processed/{name}/`."""
    directory = PROJECT_ROOT / "data" / "processed" / config["name"]
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def vocab_owner(config: dict) -> str:
    """Nombre de la corrida dueña del vocabulario que usa esta configuración.

    Normalmente es la corrida misma, pero una config puede declarar
    `vocab_from: otra_corrida` para **reutilizar** un vocabulario ya construido
    en vez de rearmarlo. Es lo que permite que una corrida piloto sobre una
    muestra entrene en el mismo espacio de índices que la corrida completa, y
    que sus resultados sean directamente comparables.
    """
    return config.get("vocab_from") or config["name"]


def reuses_vocabulary(config: dict) -> bool:
    """`True` si la config toma prestado el vocabulario de otra corrida."""
    return vocab_owner(config) != config["name"]


def checkpoint_dir(config: dict, *, create: bool = False) -> Path:
    """Directorio de checkpoints de la corrida: `checkpoints/{name}/`."""
    directory = PROJECT_ROOT / "checkpoints" / config["name"]
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def history_path(config: dict, *, create: bool = False) -> Path:
    """Historial de entrenamiento: `checkpoints/{name}/history.json`.

    Se guarda aparte de los checkpoints (que son pesados) para que el notebook
    de comparación de configuraciones pueda leer las curvas de varias corridas
    sin cargar ningún modelo.
    """
    return checkpoint_dir(config, create=create) / "history.json"


def vocab_path(config: dict, *, create: bool = False) -> Path:
    """Ruta del vocabulario: `data/processed/{vocab_owner}/vocab.json`."""
    directory = PROJECT_ROOT / "data" / "processed" / vocab_owner(config)
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory / "vocab.json"
