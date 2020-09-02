# type: ignore
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from genetic_nn.config import NUM_INPUTS, NUM_OUTPUTS, HIDDEN_LAYERS


class Genome:
    def __init__(
        self,
        weights: Optional[List[np.ndarray[np.float64]]] = None,
        biases: Optional[List[np.ndarray[np.float64]]] = None,
    ) -> None:
        self.rng = np.random.default_rng()

        self.weights: List[np.ndarray[np.float64]] = self._generate_weights() if weights is None else weights
        self.biases: List[np.ndarray[np.float64]] = self._generate_biases() if biases is None else biases

    def _generate_weights(self) -> List[np.ndarray[np.float64]]:
        sizes = (NUM_INPUTS,) + HIDDEN_LAYERS + (NUM_OUTPUTS,)

        return [self.rng.normal(size=(sizes[i], sizes[i - 1])) for i in range(1, len(sizes))]

    def _generate_biases(self) -> List[np.ndarray[np.float64]]:
        sizes = HIDDEN_LAYERS + (NUM_OUTPUTS,)

        return [self.rng.normal(size=(sizes[i],)) for i in range(len(sizes))]

    def _mutate_weights(self) -> None:
        for weight in self.weights:
            weight += self.rng.normal(size=weight.shape)

    def _mutate_biases(self) -> None:
        for bias in self.biases:
            bias += self.rng.normal(size=bias.shape)

    @staticmethod
    def _relu(x: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        return np.maximum(x, 0)

    @classmethod
    def crossover(cls, parent1: Genome, parent2: Genome) -> Tuple[Genome, Genome]:
        rng = np.random.default_rng()

        new_weights1 = []
        new_weights2 = []
        new_biases1 = []
        new_biases2 = []

        for weight1, weight2 in zip(parent1.weights, parent2.weights):
            weight1_reshaped = weight1.reshape(weight1.shape[0] * weight1.shape[1])
            weight2_reshaped = weight2.reshape(weight2.shape[0] * weight2.shape[1])

            split_point = rng.integers(1, weight1_reshaped.shape[0] - 1)

            new_weights1.append(
                np.reshape(
                    np.concatenate((weight1_reshaped[:split_point], weight2_reshaped[split_point:])), weight1.shape
                )
            )
            new_weights2.append(
                np.reshape(
                    np.concatenate((weight2_reshaped[:split_point], weight1_reshaped[split_point:])), weight2.shape
                )
            )

        for bias1, bias2 in zip(parent1.biases, parent2.biases):
            split_point = rng.integers(1, bias1.shape[0] - 1)

            new_biases1.append(np.concatenate((bias1[:split_point], bias2[split_point:])))
            new_biases2.append(np.concatenate((bias2[:split_point], bias1[split_point:])))

        return (cls(new_weights1, new_biases1), cls(new_weights2, new_biases2))

    def evaluate(self, inputs: Union[np.ndarray[np.float64], Sequence[float]]) -> np.array[np.float64]:
        assert np.array(inputs).shape == (
            NUM_INPUTS,
        ), "Number of inputs given does not match number of inputs specified"

        for idx, weight in enumerate(self.weights):
            inputs = self._relu((weight @ inputs) + self.biases[idx])

        return inputs

    def mutate(self) -> None:
        if self.rng.random() < 0.3:
            self._mutate_weights()

        if self.rng.random() < 0.3:
            self._mutate_biases()
