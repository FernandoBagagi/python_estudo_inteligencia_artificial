class No:

    def __init__(self, estado, no_pai, custo):
        self.estado = estado  # Estado atual
        self.NoPai = no_pai  # De onde veio
        self.custo = custo  # 1

    def teste_objetivo(self):
        return self.estado.isObjetivo()

    def get_sucessores(self):
        resultado = []
        l_estados = self.estado.sucessor()
        for s in l_estados:
            no = No(s, self, 1)
            resultado.append(no)
        return resultado
