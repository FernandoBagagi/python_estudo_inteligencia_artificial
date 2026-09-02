import numpy as np
import matplotlib.pyplot as plt
from math import sin

# Min e Max da função
# v_min, v_max = -6.0, 6.0
v_min, v_max = 2.7, 7.5 

def objetivo(x):
    #return x**2.0
    return sin(x) + sin((10/3)*x)

# def amostra(x_min, x_max):
#     X = np.arange(x_min, x_max, 0.1)
#     # Aplicando a função objetivo nas entradas
#     Y = [objetivo(x) for x in X]
#     return X, Y

# X, Y = amostra(v_min, v_max)

# plt.plot(X, Y)
# plt.axvline(x=5.1457, ls='--', color='g') # "0" or ""
# plt.show()

def sucessor(x, l_min, l_max):
    
    x_delta = np.random.uniform(-1, 1)
    x_new   = x + x_delta
    
    if x_new > l_max:
        return l_max
    
    if x_new < l_min:
        return l_min
    
    return x_new 

#plt.plot(X, Y)

# Estado Inicial
x_atual = 3
y_atual = objetivo(x_atual)
plt.scatter(x_atual, y_atual, s=20, color = "green")

# Limite para o Loop
limit_total = 1000
n_total = 0
limit_sem_melhora = 50
n_sem_melhora = 0

while True:    
    x_proximo = sucessor(x_atual, v_min, v_max)
    y_proximo = objetivo(x_proximo)
    if y_proximo < y_atual:
        x_atual = x_proximo
        y_atual = y_proximo    
        plt.scatter(x_atual, y_atual, s=20, color = "green")
    else:
        n_sem_melhora = n_sem_melhora + 1
    n_total = n_total + 1
    if n_total > limit_total:
        print('Parou pelo limite total')
        break
    if n_sem_melhora > limit_sem_melhora:
        print('Parou pelo limite sem melhora')
        print('X: ' + str(x_atual))
        break

plt.scatter(x_atual, y_atual, s=50, color = "yellow")        
plt.show()