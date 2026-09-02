from Matriz import Matriz
from Borda import Borda
#import sys
#sys.setrecursionlimit(1000000000)
#print(sys.getrecursionlimit())

#b = Borda([0,1,2,3,4,5,6,7,8])
b = Borda([3,1,2,6,4,5,7,8,0])
#b = Borda([3,1,2,6,4,5,7,8,0])
#b = Borda([0,1,4,6,3,7,8,2,5])
b.solucionar()
"""while not(b.solucionar()):
    pass"""

"""m = Matriz(1,0,2,3,4,5,6,7,8)
m.imprimir_todos()
print(m.localizar_zero())
#print(m.imprimir_adjacentes([0,1]))
print(m.get_adjacentes([0,1]))
for m_aux in m.get_sucessores():
    m_aux.imprimir_todos()
    if m_aux.concluido():
        print('Terminou')
m2 = Matriz(1,0,2,3,4,5,6,7,8)
m3 = Matriz(1,0,2,3,4,5,6,7,8)
print(m3 in [m, m, m])
print(m == m3)"""


#resultados = m.solucionar()
#print(resultados[0].solucionar())
"""if type(m.solucionar()) == list:
    print(m.solucionar())
print(m.solucionar())
m.imprimir_todos()
m.trocar(m.localizar(),[1,1])
m.imprimir_todos()
print(m.adjacentes(m.localizar()))
m.imprimir_adjacentes(m.localizar())
t = m.clonar()
t.imprimir_todos()"""