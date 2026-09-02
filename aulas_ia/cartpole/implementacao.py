# fonte https://www.youtube.com/watch?v=JNKvJEzuNsc&ab_channel=RichardBrooker

class Metricas:

    def __init__(self, epoca, tempo, aprendizado):
        self.epoca = epoca
        self.tempo = tempo
        self.aprendizado = aprendizado


"""
Os dados de entrada são:
Posição do carrinho (cart position) -4.8 ~ 4.8
Velocidade do carrinho (cart velocity) -inf ~ +inf
Ângulo do poste (pole angle) -24 deg ~ 24 deg
Velocidade do poste (pole velocity) -inf ~ +inf

recompensa é 1 se o angulo for menor que 12 e 0 se for maior ou igual

Ações:
Mover para a esquerda (Push cart to the left)
Mover para a direira (Push cart to the right)
"""

# Importações
from unicodedata import name
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import time
import math
import random
from typing import Tuple
import gym

# Seleciona o jogo
env = gym.make('CartPole-v1')

# 0 move para a esquerda e 1 move para a direita


def policy(obs): return 1  # constante


def policy(_, __, ___, tip_velocity): return int(tip_velocity > 0)

'''
for _ in range(5):
    obs = env.reset()
    for _ in range(80):
        actions = policy(*obs)
        obs, reward, done, info = env.step(actions) 
        env.render()
        time.sleep(0.05)

env.close()
'''


# Discretização
n_bins = (6, 12)
# [-0.41887903, -0.8726646259971648]
lower_bounds = [env.observation_space.low[2], -math.radians(50)]
# [ 0.41887903,  0.8726646259971648]
upper_bounds = [env.observation_space.high[2], math.radians(50)]


def discretizer(_, __, angle, pole_velocity) -> Tuple[int, ...]:
    """Convert continues state intro a discrete state"""
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    est.fit([lower_bounds, upper_bounds])
    return tuple(map(int, est.transform([[angle, pole_velocity]])[0]))


# Inicializa a tabela Q com zeros
Q_table = np.zeros(n_bins + (env.action_space.n,))
Q_table.shape

# A política é escolher baaseado no maior valor da tabela


def policy(state: tuple):
    """Choosing action based on epsilon-greedy policy"""
    return np.argmax(Q_table[state])


def new_Q_value(reward: float,  new_state: tuple, discount_factor=1) -> float:
    """Temperal diffrence for updating Q-value of state-action pair"""
    future_optimal_value = np.max(Q_table[new_state])
    learned_value = reward + discount_factor * future_optimal_value
    return learned_value


# Adaptive learning of Learning Rate
def learning_rate(n: int, min_rate=0.01) -> float:
    """Decaying learning rate"""
    return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))

# Aleatoriedade no aprendizado


def exploration_rate(n: int, min_rate=0.1) -> float:
    """Decaying exploration rate"""
    return max(min_rate, min(1, 1.0 - math.log10((n + 1) / 25)))


# Treinamento
metricas = []
n_episodes = 200
for e in range(n_episodes):

    comeco = time.process_time()
    acoes = []

    # Siscretize state into buckets
    current_state, done = discretizer(*env.reset()), False

    while done == False:

        # policy action
        action = policy(current_state)  # exploit

        # insert random action
        if np.random.random() < exploration_rate(e):
            action = env.action_space.sample()  # explore

        acoes.append(action)

        # increment enviroment
        obs, reward, done, _ = env.step(action)
        #print('A recompensa foi de ', reward)
        new_state = discretizer(*obs)

        # Update Q-Table
        lr = learning_rate(e)
        learnt_value = new_Q_value(reward, new_state)
        old_value = Q_table[current_state][action]
        Q_table[current_state][action] = (1-lr)*old_value + lr*learnt_value

        current_state = new_state

        # Render the cartpole environment

        env.render()

    metricas.append(Metricas(epoca=e, tempo=time.process_time() - comeco, aprendizado=learning_rate(e)))


import matplotlib.pyplot as plt

x = [aux.epoca for aux in metricas]
y1 = [aux.tempo for aux in metricas]
y2 = [aux.aprendizado for aux in metricas]

plt.plot(x, y1)
plt.title('Tempo de duração')
plt.xlabel('Épocas')
plt.ylabel('Tempo (sgundos)')

plt.show()