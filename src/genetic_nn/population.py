# type: ignore
from copy import deepcopy
from typing import Callable, List, Tuple

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

    def evaluate_population(self, eval_func: Callable[[Genome], float]) -> Tuple[np.ndarray, Genome]:
        self.fitnesses = np.zeros((POPULATION,))

        for idx, genome in enumerate(self.genomes):
            self.fitnesses[idx] = eval_func(genome)

        return self.fitnesses, self.genomes[np.argmax(self.fitnesses)]

    def breed_next_generation(self) -> None:
        probabilities: np.ndarray[np.float64] = np.maximum(self.fitnesses, 1) / np.maximum(self.fitnesses, 1).sum()
        new_generation: List[Genome] = []

        for _ in range(self.rng.integers((POPULATION - 5) // 2, (POPULATION - 3) // 2)):
            parent1, parent2 = self.rng.choice(self.genomes, size=2, replace=False, p=probabilities)
            new_generation.extend(Genome.crossover(parent1, parent2))

        while len(new_generation) < POPULATION - 2:
            if self.rng.random() < 0.005:
                new_generation.append(Genome())
            else:
                new_generation.append(deepcopy(self.rng.choice(self.genomes, size=1, p=probabilities)[0]))

        for genome in new_generation:
            genome.mutate()

        for idx in np.argpartition(self.fitnesses, -2)[-2:]:
            new_generation.append(deepcopy(self.genomes[idx]))

        self.genomes = new_generation
