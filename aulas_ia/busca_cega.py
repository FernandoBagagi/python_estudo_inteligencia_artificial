class No:

    def __init__(self, estado, no_pai, custo):
        self.estado = estado
        self.no_pai = no_pai
        self.custo = custo

    def concluiu(self):
        return self.estado.concluido()

    def get_sucessores(self):
        sucessores = []
        for matriz in self.estado.get_sucessores():
            sucessores.append(No(matriz, self, 1))
        return sucessores

    def __eq__(self, __o):
        return self.estado == __o.estado

    def mostrar_caminho(self):
        lista_nos = []
        no_atual = self
        while no_atual:
            lista_nos.append(no_atual)
            no_atual = no_atual.no_pai
        for no_aux in lista_nos[::-1]:
            no_aux.estado.imprimir_todos()

class Matriz:

    def __init__(self, *args):
        self.matriz = [
            list(args[:3]),
            list(args[3:6]),
            list(args[6:]),
        ]

    def imprimir_todos(self):
        for linha in range(3):
            for coluna in range(3):
                print(self.matriz[linha][coluna], end=' ')
            print()
        print()

    def localizar_zero(self):
        for linha in range(3):
            for coluna in range(3):
                if self.matriz[linha][coluna] == False:
                    return [linha,coluna]

    def get_adjacentes(self, posicao):
        adjacentes = []
        if posicao[1] - 1 >=0:
            adjacentes.append([posicao[0],posicao[1]-1])
        if posicao[0] - 1 >=0:
            adjacentes.append([posicao[0] - 1,posicao[1]])
        if posicao[0] + 1 < 3:
            adjacentes.append([posicao[0]+1,posicao[1]])
        if posicao[1] + 1 < 3:
            adjacentes.append([posicao[0],posicao[1]+1])
        return adjacentes

    def get_sucessores(self):
        sucessores = []
        local_zero = self.localizar_zero()
        for i in self.get_adjacentes(local_zero):
            sucessores.append(self.clonar_mudando(local_zero,i))
        return sucessores

    def imprimir_adjacentes(self, posicao):
        for posicao_aux in self.get_adjacentes(posicao):
            print(self.matriz[posicao_aux[0]][posicao_aux[1]])
        
    def trocar(self,posicao1,posicao2):
        aux = self.matriz[posicao1[0]][posicao1[1]]
        self.matriz[posicao1[0]][posicao1[1]] = self.matriz[posicao2[0]][posicao2[1]]
        self.matriz[posicao2[0]][posicao2[1]] = aux

    def clonar(self):
        aux = []
        for i in range(3):
            aux.extend(self.matriz[i])
        return Matriz(*aux)

    def clonar_mudando(self,posicao1,posicao2):
        matriz_aux = self.clonar()
        matriz_aux.trocar(posicao1,posicao2)
        return matriz_aux

    def concluido(self):
        return self.matriz == [[0,1,2],[3,4,5],[6,7,8]]

    def __eq__(self, __o):
        return self.matriz == __o.matriz
    
class Borda:

    visitados = []

    nao_visitados = []

    def __init__(self, lista):
        self.nao_visitados.append(No(Matriz(*lista), None, 1))

    """def solucionar(self):
        if not(self.get_novas_possibilidades(self.nao_visitados[0])):
            self.visitados.append(self.nao_visitados.pop(0))
            #del(self.nao_visitados[0])
            #print(len(self.visitados))
            #print(len(self.nao_visitados))
            self.solucionar()"""

    def solucionar(self):
        while self.get_novas_possibilidades(self.nao_visitados[0]):
            print(f'Foram visitados {len(self.visitados)} nós!')
            self.visitados.append(self.nao_visitados.pop(0))


    def get_novas_possibilidades(self, no):
        if no.concluiu():
            print(f'Concluiu!') 
            print(f'Foram visitados {len(self.visitados)} nós!')
            print(f'Faltou visitar {len(self.nao_visitados)} nós!')
            no.mostrar_caminho()
            return False
        else:
            for sucessor in no.get_sucessores():
                if not(sucessor in self.visitados):
                    if not(sucessor in self.nao_visitados):
                        self.nao_visitados.append(sucessor)

        return True

import random as rnd

lista = [0,1,2,3,4,5,6,7,8]
rnd.shuffle(lista)
print(lista)
b = Borda(lista)
#b = Borda([3,1,2,6,4,5,7,8,0])
#b = Borda([1,2,5,0,3,4,6,7,8])
b.solucionar()