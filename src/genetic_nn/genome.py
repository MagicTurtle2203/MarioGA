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

            split_point = rng.randint(1, weight1_reshaped.shape[0])

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
            bias1_reshaped = bias1.reshape(bias1.shape[0] * bias1.shape[1])
            bias2_reshaped = bias2.reshape(bias2.shape[0] * bias2.shape[1])

            split_point = rng.randint(1, bias1_reshaped.shape[0])

            new_biases1.append(
                np.reshape(np.concatenate((bias1_reshaped[:split_point], bias2_reshaped[split_point:])), bias1.shape)
            )
            new_biases2.append(
                np.reshape(np.concatenate((bias2_reshaped[:split_point], bias1_reshaped[split_point:])), bias2.shape)
            )

        return (cls(new_weights1, new_biases1), cls(new_weights2, new_biases2))

    def evaluate(self, inputs: Union[np.ndarray[np.float64], Sequence[np.float64]]) -> int:
        assert np.array(inputs).shape == (
            NUM_INPUTS,
        ), "Number of inputs given does not match number of inputs specified"

        for idx, weight in enumerate(self.weights):
            inputs = self._relu((weight @ inputs) + self.biases[idx])

        return np.argmax(inputs)

    def mutate(self) -> None:
        if self.rng.random() < 0.8:
            self._mutate_weights()

        if self.rng.random() < 0.8:
            self._mutate_biases()
