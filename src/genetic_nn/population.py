# type: ignore
from typing import Callable, List

import numpy as np

from genetic_nn.genome import Genome
from genetic_nn.config import POPULATION


class Population:
    def __init__(self) -> None:
        self.rng = np.random.default_rng()

        self.genomes: List[Genome] = []
        self.fitnesses: np.ndarray[np.float64] = np.zeros((POPULATION,))

        for _ in range(POPULATION):
            self.genomes.append(Genome())

    def evaluate_population(self, eval_func: Callable[[Genome], float]) -> np.ndarray:
        self.fitnesses = np.zeros((POPULATION,))

        for idx, genome in enumerate(self.genomes):
            self.fitnesses[idx] = eval_func(genome)

        return self.fitnesses

    def breed_next_generation(self) -> None:
        probabilities: np.ndarray[np.float64] = np.maximum(self.fitnesses, 1) / np.maximum(self.fitnesses, 1).sum()
        new_generation: List[Genome] = []

        for _ in range(self.rng.integers((POPULATION // 2) - 3, POPULATION // 2)):
            if self.rng.random() < 0.05:
                new_generation.extend((Genome(), Genome()))
            else:
                parent1, parent2 = self.rng.choice(self.genomes, size=2, replace=False, p=probabilities)
                new_generation.extend(Genome.crossover(parent1, parent2))

        while len(new_generation) < POPULATION:
            new_generation.append(self.rng.choice(self.genomes, size=1, p=probabilities)[0])

        for genome in new_generation:
            genome.mutate()

        self.genomes = new_generation
