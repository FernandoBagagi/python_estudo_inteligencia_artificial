import bitstring
import random

taxa_crossover = 0.9
taxa_mutacao = 0.01

possibilidades = list(range(-10,11,1))

# Primeiramente cria o vetor da população
populacao_inicial = []

# Seleciona aleatóriamente 4 indivíduos para serem os primeiros da população
populacao_inicial.append(possibilidades.pop(random.randint(0, len(possibilidades)-1)))
populacao_inicial.append(possibilidades.pop(random.randint(0, len(possibilidades)-1)))
populacao_inicial.append(possibilidades.pop(random.randint(0, len(possibilidades)-1)))
populacao_inicial.append(possibilidades.pop(random.randint(0, len(possibilidades)-1)))

# Mostra os números escolhidos em decimal e em binário
print('Escolhendo a populacao inicial')
for paux in populacao_inicial:
    print(str(paux) + ' = ' + str(bitstring.BitArray('int:5=' + str(paux))))


class Genoma:
    # A classe genoma encapsula o cromossomo e sua aptidão

    def __init__(self, dado):
        self.cromossomo = bitstring.BitArray('int:5=' + str(dado))
        self.aptidao = self.__calcularAptidao(dado)

    def __calcularAptidao(self, dado):
        return (dado ** 2) - (3 * dado) + 4

    def recalcularAptidao(self):
        self.aptidao = self.__calcularAptidao(self.cromossomo.int)

    def __str__(self) -> str:
        return 'cromossomo: ' + str(self.cromossomo) + ' aptidao: ' + str(self.aptidao)

    def __eq__(self, __o: object) -> bool:
        return self.aptidao == __o.aptidao and self.cromossomo.int == __o.cromossomo.int

    def __lt__(self, __o: object) -> bool:
        return self.aptidao < __o.aptidao


# Transformando a população inicial em genoma
populacao_genoma = list(map(lambda item: Genoma(item), populacao_inicial))

# Mostra a população transformada em genoma
print('\nTransformando em genoma:')
for paux in populacao_genoma:
    print(paux)


def torneio(lista_candidatos):
    # Sorteia os indices
    # Com os indices sorteados preenche as listas do torneio
    candidatos_1 = []
    candidatos_1.append(lista_candidatos.pop(random.randint(0, len(lista_candidatos) - 1)))
    candidatos_1.append(lista_candidatos.pop(random.randint(0, len(lista_candidatos) - 1)))
    candidatos_2 = lista_candidatos

    print(' Candidatos 1:')
    for aux in candidatos_1:
        print('  ', aux)

    print(' Candidatos 2:')
    for aux in candidatos_2:
        print('  ', aux)

    # Faz o torneio
    resultado = []
    resultado.append(
        candidatos_1[0] if candidatos_1[0] < candidatos_1[1] else candidatos_1[1])
    resultado.append(
        candidatos_2[0] if candidatos_2[0] < candidatos_2[1] else candidatos_2[1])

    return resultado


populacao_torneio = torneio(populacao_genoma)

print('\nO resultado do torneio foi:')
print(populacao_torneio[0])
print(populacao_torneio[1])

print('Iniciando algoritmo\n\n')
limite_interacoes = 20
limite_sem_mudancas = 5
melhor_gene = Genoma(14)
while limite_interacoes and limite_sem_mudancas:
    melhor_gene_temp = populacao_torneio[0] if populacao_torneio[0] < populacao_torneio[1] else populacao_torneio[1]
    if melhor_gene == melhor_gene_temp:
        print('\nNao melhorou o gene')
        limite_sem_mudancas -= 1
    else:
        if melhor_gene_temp < melhor_gene:
            print('\nMelhorou o gene')
            limite_sem_mudancas = 5
            melhor_gene = melhor_gene_temp
    
    filho1 = populacao_torneio[0]
    filho2 = populacao_torneio[1]
    #crossover
    if random.random() <= taxa_crossover:
        print('\nIniciando crossover')
        particao = random.randint(1,4)
        filho1 = Genoma((populacao_torneio[0].cromossomo[0:particao] + populacao_torneio[1].cromossomo[particao:5]).int)
        filho2 = Genoma((populacao_torneio[1].cromossomo[0:particao] + populacao_torneio[0].cromossomo[particao:5]).int)

        
    populacao_torneio.append(filho1)
    populacao_torneio.append(filho2)
    #Mutação
    for gene in populacao_torneio:
        for i in range(0,5):
            if random.random() <= taxa_mutacao:
                gene.cromossomo[i] = 0 if gene.cromossomo[i] else 1
                gene.recalcularAptidao()
                print('\n\nHouve mutacao!!!\n')

    populacao_torneio = torneio(populacao_torneio)

    print('\nO resultado do torneio foi:')
    print(populacao_torneio[0])
    print(populacao_torneio[1])

    limite_interacoes -= 1

r = populacao_torneio[0] if populacao_torneio[0] < populacao_torneio[1] else populacao_torneio[1]
print('Convergiu para ', r.cromossomo.int)
lista = list(range(-10,10))
lista = list(map(lambda x: (x ** 2) - (3 * x) + 4, lista))
print('Sendo o min da funcao: ', min(lista))