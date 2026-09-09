"""Fase 4 — Arquitectura CBOW en PyTorch con negative sampling.

El modelo tiene **dos** matrices de embeddings, no una:

* `input_embeddings` — la representación de una palabra cuando aparece *como
  contexto*. Es la que se exporta al final: son "los embeddings".
* `output_embeddings` — la representación de una palabra cuando aparece *como
  target*. Solo se usa para puntuar durante el entrenamiento y después se
  descarta.

El camino de un batch es:

    contexto (B, 2C)  --input_embeddings-->  (B, 2C, D)
                      --promedio enmascarado-->  proyección (B, D)
    proyección · output_embeddings[target]     -> puntaje positivo (B,)
    proyección · output_embeddings[negativos]  -> puntajes negativos (B, K)

La capa de salida usa **negative sampling** en vez de softmax completo: en vez
de calcular un puntaje contra las 5.000 palabras del vocabulario en cada paso,
se calcula contra la palabra correcta y `K` palabras sorteadas al azar. Con
vocabulario 5.000 y K=10 eso es 11 productos en vez de 5.000; al escalar a
vocabulario 20.000 la diferencia es de casi tres órdenes de magnitud.
"""

from __future__ import annotations

import argparse
import math
from typing import Sequence

import torch
from torch import nn

from src import configure_console
from src.config import load_config, vocab_path
from src.vocabulary import Vocabulary

__all__ = [
    "masked_mean",
    "NegativeSampler",
    "CBOWModel",
    "build_model_from_config",
]


def masked_mean(vectors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Promedia `vectors` sobre la dimensión del contexto, ignorando el relleno.

    Es la operación central de CBOW ("bag of words": el contexto es un conjunto
    sin orden, y se lo resume promediando). La máscara de la Fase 3 hace que las
    posiciones de relleno no cuenten **ni en la suma ni en el divisor**: sin eso,
    un contexto de 2 palabras rellenado a 4 se dividiría por 4 y el vector
    saldría a la mitad.

    Args:
        vectors: `(B, L, D)` — los embeddings de cada posición del contexto.
        mask: `(B, L)` — 1 en las posiciones reales, 0 en el relleno.

    Returns:
        `(B, D)` con el promedio de las posiciones reales de cada ejemplo.
    """
    weights = mask.unsqueeze(-1).to(vectors.dtype)          # (B, L, 1)
    total = (vectors * weights).sum(dim=1)                  # (B, D)
    counts = weights.sum(dim=1).clamp(min=1.0)              # (B, 1), nunca 0
    return total / counts


class NegativeSampler(nn.Module):
    """Sorteo de palabras negativas según la distribución unigrama elevada a 3/4.

    Mikolov et al. observaron que muestrear los negativos proporcionalmente a
    `frecuencia^0.75` funciona mejor que hacerlo proporcional a la frecuencia
    (que sortearía "de" y "la" casi siempre, negativos demasiado fáciles) o
    uniformemente (que sortearía palabras rarísimas, negativos poco informativos).
    El exponente 3/4 aplana la distribución: le baja el peso a las muy frecuentes
    sin llegar a igualar a todas.

    Es un `nn.Module` para que la tabla de sorteo viaje sola a la GPU con
    `.to(device)`. La tabla no se guarda en los checkpoints (`persistent=False`)
    porque se puede reconstruir del vocabulario.
    """

    def __init__(
        self,
        counts: Sequence[int],
        *,
        power: float = 0.75,
        exclude: Sequence[int] = (0,),
        seed: int | None = None,
    ) -> None:
        super().__init__()
        weights = torch.tensor(counts, dtype=torch.double).clamp(min=0.0) ** power
        # `<UNK>` se excluye por defecto: con `drop_unknown=True` nunca aparece
        # como target, así que tampoco tiene sentido usarlo como negativo.
        for index in exclude:
            weights[index] = 0.0
        if weights.sum() <= 0:
            raise ValueError("todos los pesos de muestreo quedaron en cero")

        probabilities = (weights / weights.sum()).to(torch.float32)
        self.register_buffer("probabilities", probabilities, persistent=False)
        # Muestrear = sortear u ~ U(0,1) y buscar dónde cae en la acumulada.
        # `searchsorted` lo hace en O(log V) y totalmente vectorizado.
        self.register_buffer("cdf", probabilities.double().cumsum(0), persistent=False)

        self.vocab_size = len(probabilities)
        self.generator: torch.Generator | None = None
        if seed is not None:
            self.generator = torch.Generator().manual_seed(seed)

    def sample(self, batch_size: int, k: int) -> torch.Tensor:
        """Sortea `(batch_size, k)` índices de palabras negativas.

        Se sortea con reemplazo y **sin** excluir al target positivo del batch:
        con un vocabulario de miles de palabras la colisión es rarísima, y es lo
        que hace el word2vec original.
        """
        device = self.cdf.device
        if self.generator is not None:
            uniform = torch.rand(
                batch_size * k, generator=self.generator, dtype=torch.double
            ).to(device)
        else:
            uniform = torch.rand(batch_size * k, dtype=torch.double, device=device)
        indices = torch.searchsorted(self.cdf, uniform)
        return indices.clamp_(max=self.vocab_size - 1).view(batch_size, k)

    def __repr__(self) -> str:
        return f"NegativeSampler(vocab_size={self.vocab_size}, power=0.75)"


class CBOWModel(nn.Module):
    """Continuous Bag of Words con negative sampling.

    Args:
        vocab_size: tamaño del vocabulario (incluye `<UNK>`).
        embedding_dim: dimensión de los vectores.
        seed: semilla para la inicialización, para que sea reproducible.
    """

    def __init__(
        self, vocab_size: int, embedding_dim: int, *, seed: int | None = None
    ) -> None:
        super().__init__()
        if vocab_size < 2:
            raise ValueError(f"vocab_size debe ser >= 2, no {vocab_size!r}")
        if embedding_dim < 1:
            raise ValueError(f"embedding_dim debe ser >= 1, no {embedding_dim!r}")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.reset_parameters(seed)

    def reset_parameters(self, seed: int | None = None) -> None:
        """Inicialización del word2vec original.

        Los embeddings de entrada arrancan con ruido chico y uniforme; los de
        salida arrancan en **cero**. Eso hace que todos los puntajes iniciales
        valgan 0, la sigmoide dé 0,5 y la loss inicial sea exactamente
        `(1 + K) · ln 2` — un valor conocido, muy útil para verificar de un
        vistazo que el modelo arrancó bien.
        """
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)

        bound = 0.5 / self.embedding_dim
        with torch.no_grad():
            self.input_embeddings.weight.uniform_(-bound, bound, generator=generator)
            self.output_embeddings.weight.zero_()

    # -- pasos del forward, separados para poder inspeccionarlos ----------

    def forward(self, context: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Proyecta un batch de contextos al espacio de embeddings.

        Args:
            context: `(B, 2C)` índices de las palabras de contexto.
            mask: `(B, 2C)` 1 en las posiciones reales, 0 en el relleno.

        Returns:
            `(B, D)` — el vector que resume cada contexto.
        """
        vectors = self.input_embeddings(context)        # (B, 2C, D)
        return masked_mean(vectors, mask)               # (B, D)

    def score(self, projection: torch.Tensor, words: torch.Tensor) -> torch.Tensor:
        """Puntaje (producto punto) entre cada proyección y palabras candidatas.

        Args:
            projection: `(B, D)`.
            words: `(B,)` para una palabra por ejemplo, o `(B, K)` para varias.

        Returns:
            `(B,)` o `(B, K)` según la forma de `words`.
        """
        vectors = self.output_embeddings(words)
        if words.dim() == 1:
            return (projection * vectors).sum(dim=-1)               # (B,)
        return torch.bmm(vectors, projection.unsqueeze(-1)).squeeze(-1)  # (B, K)

    def negative_sampling_loss(
        self,
        context: torch.Tensor,
        mask: torch.Tensor,
        target: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """Loss de negative sampling, promediada sobre el batch.

        Para cada ejemplo se maximiza la probabilidad de que el target sea el
        correcto y se minimiza la de las `K` palabras sorteadas:

            L = -log σ(h · v_target) - Σ_k log σ(-h · v_negativo_k)

        Se usa `logsigmoid` en vez de `log(sigmoid(x))` porque es numéricamente
        estable con puntajes grandes en valor absoluto.

        Args:
            context: `(B, 2C)`; mask: `(B, 2C)`; target: `(B,)`;
            negatives: `(B, K)`.

        Returns:
            Escalar.
        """
        projection = self.forward(context, mask)                    # (B, D)
        positive = self.score(projection, target)                   # (B,)
        negative = self.score(projection, negatives)                # (B, K)

        positive_loss = -nn.functional.logsigmoid(positive)          # (B,)
        negative_loss = -nn.functional.logsigmoid(-negative).sum(dim=1)  # (B,)
        return (positive_loss + negative_loss).mean()

    def full_softmax_logits(
        self, context: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Puntajes contra **todo** el vocabulario: `(B, V)`.

        No se usa para entrenar —es justamente lo que el negative sampling
        evita—, pero sirve para comparar los dos enfoques en el notebook y para
        inspeccionar qué predice el modelo.
        """
        projection = self.forward(context, mask)
        return projection @ self.output_embeddings.weight.T

    # -- utilidades ------------------------------------------------------

    @property
    def embeddings(self) -> torch.Tensor:
        """La matriz que se exporta al final: `(V, D)`."""
        return self.input_embeddings.weight

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"CBOWModel(vocab_size={self.vocab_size}, "
            f"embedding_dim={self.embedding_dim}, "
            f"parámetros={self.n_parameters():,})"
        )


def build_model_from_config(
    config: dict, vocab: Vocabulary
) -> tuple[CBOWModel, NegativeSampler]:
    """Arma el modelo y el muestreador a partir de un YAML y su vocabulario.

    El tamaño del modelo sale del vocabulario real, no del `vocab_size` del
    YAML: si el vocabulario quedó más chico (por `min_count`), la matriz tiene
    que coincidir con él o los índices no cierran.
    """
    seed = config.get("seed", 42)
    model = CBOWModel(len(vocab), config["embedding_dim"], seed=seed)
    sampler = NegativeSampler(vocab.counts, seed=seed)
    return model, sampler


def _main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(description="Inspecciona el modelo CBOW.")
    parser.add_argument("--config", default="configs/piloto_5k_50_2.yaml")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config = load_config(args.config)
    vocab = Vocabulary.load(vocab_path(config))
    model, sampler = build_model_from_config(config, vocab)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    sampler.to(device)

    context_width = 2 * config["context_size"]
    k = config["negative_samples"]
    batch = args.batch_size

    context = torch.randint(0, len(vocab), (batch, context_width), device=device)
    mask = torch.ones(batch, context_width, device=device)
    target = torch.randint(0, len(vocab), (batch,), device=device)
    negatives = sampler.sample(batch, k)

    projection = model(context, mask)
    loss = model.negative_sampling_loss(context, mask, target, negatives)

    print(f"config     : {config['name']} | dispositivo: {device}")
    print(f"modelo     : {model}")
    print(f"muestreador: {sampler}")
    print()
    print(f"contexto   : {tuple(context.shape)}")
    print(f"máscara    : {tuple(mask.shape)}")
    print(f"proyección : {tuple(projection.shape)}")
    print(f"negativos  : {tuple(negatives.shape)}")
    print(f"loss       : {loss.item():.4f}")
    print(f"esperada   : {(1 + k) * math.log(2):.4f}  ((1+K)·ln2 al inicializar)")


if __name__ == "__main__":
    _main()
