from No import No
from Matriz import Matriz

class Borda:

    visitados = []

    nao_visitados = []

    def __init__(self, lista):
        self.nao_visitados.append(No(Matriz(*lista), None, 1))

    def solucionar(self):
        while self.get_novas_possibilidades(self.nao_visitados[0]):
            print(f'Foram visitados {len(self.visitados)} nós!')
            self.visitados.append(self.nao_visitados.pop(0))
        """if self.get_novas_possibilidades(self.nao_visitados[0]):
            self.visitados.append(self.nao_visitados.pop(0))
            #del(self.nao_visitados[0])
            #print(len(self.visitados))
            #print(len(self.nao_visitados))
            self.solucionar()"""


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


