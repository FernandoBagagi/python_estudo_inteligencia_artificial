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

    """def solucionar(self):
        if self.concluido():
            self.imprimir_todos()
            print('Terminou')
            sys.exit()
        branco = self.localizar_zero()
        self.imprimir_todos()
        #print(branco)
        adjacentes = self.get_adjacentes(branco)
        estados = []
        #self.imprimir_adjacentes(self.localizar())
        for i in range(len(adjacentes)):
            aux = self.clonar()
            aux.trocar(branco,adjacentes[i])
            if aux.concluido():
                aux.imprimir_todos()
                print('Terminou')
                sys.exit()
            else:
                estados.append(aux.clonar())
        return estados
    """
