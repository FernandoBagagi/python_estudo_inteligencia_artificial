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
        self.estado == __o.estado

    def mostrar_caminho(self):
        lista_nos = []
        no_atual = self
        while no_atual:
            lista_nos.append(no_atual)
            no_atual = no_atual.no_pai
        for no_aux in lista_nos[::-1]:
            no_aux.estado.imprimir_todos()