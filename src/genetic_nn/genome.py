from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from genetic_nn.config import NUM_INPUTS, NUM_OUTPUTS, HIDDEN_LAYERS


class Genome:
    def __init__(
        self, weights: Optional[List[np.ndarray[float]]] = None, biases: Optional[List[np.ndarray[float]]] = None
    ) -> None:
        self.weights: List[np.ndarray[float]] = self._generate_weights() if weights is None else weights
        self.biases: List[np.ndarray[float]] = self._generate_biases() if biases is None else biases

        self.rng = np.random.default_rng()

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

    def mutate(self) -> None:
        if self.rng.random() < 0.8:
            self._mutate_weights()

        if self.rng.random() < 0.8:
            self._mutate_biases()
