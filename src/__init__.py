"""Código reutilizable del proyecto CBOW sobre el Spanish Billion Word Corpus."""

from __future__ import annotations

import sys

__all__ = ["configure_console"]


def configure_console() -> None:
    """Fuerza UTF-8 en la salida estándar, si el intérprete lo permite.

    La consola de Windows usa cp1252 por defecto y lanza `UnicodeEncodeError`
    ante cualquier carácter fuera de esa página: una flecha `→` en un mensaje de
    progreso alcanza para matar un entrenamiento de horas. Pasó dos veces durante
    el desarrollo, así que todos los CLI del proyecto llaman a esto al arrancar.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
