# type: ignore
import numpy as np
import pickle
from skimage.transform import resize

import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

from genetic_nn.population import Population


def rgb2dec(rgb) -> int:
    return (rgb[0] * 256 * 256) + (rgb[1] * 256) + rgb[2]


def eval_genome(genome):
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    done = False
    timeout = 100

    state = env.reset()

    rewards = 0

    while not done and timeout > 0:
        state_resized = resize(state, (state.shape[0] // 8, state.shape[1] // 8), anti_aliasing=False)
        state_resized = np.apply_along_axis(
            rgb2dec,
            1,
            (np.reshape(state_resized, (state_resized.shape[0] * state_resized.shape[1], 3)) * 255),
        )

        state, reward, done, info = env.step(np.argmax(genome.evaluate(state_resized)))

        rewards += reward

        if reward <= 0:
            timeout -= 1
        else:
            timeout += 1

        env.render()

    env.close()

    return rewards


if __name__ == "__main__":
    p = Population()
    all_fitnesses = []
    for i in range(50):
        print(f"Generation {i+1}")
        fitnesses, best_genome = p.evaluate_population(eval_genome)
        print(f"\tBest fitness: {fitnesses.max()}")
        print(f"\tAverage fitness: {fitnesses.sum() / fitnesses.shape[0]:.3f}")
        print(f"\tAll fitnesses: {fitnesses}")
        all_fitnesses.append(fitnesses)
        with open(fr"best_models\generation_{i+1}_best_genome.pickle", "wb") as file:
            pickle.dump(best_genome, file)
        p.breed_next_generation()
    with open(r"best_models\fitnesses.pickle", "wb") as file:
        pickle.dump(best_genome, file)
