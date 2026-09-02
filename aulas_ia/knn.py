def distanciaEuclidiana(v1, v2):
    #A distância euclidiana é x = ((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2)^(1/2)
    acumulador = 0.0
    for i in range(0, len(v1)):
      acumulador += (v1[i] - v2[i]) ** 2
    return acumulador ** (1/2)

def getListaDistanciaOrdendada(X, Y, novoExemplo):
    lista_distancia = []
    for i in range(0, len(X)):
      campo = (distanciaEuclidiana(X[i], novoExemplo), Y[i])
      lista_distancia.append(campo)
    lista_distancia.sort(key = lambda elemento: elemento[0])
    return lista_distancia

def getListaKVizinhosMaisProximos(distancias, K):
    return distancias[0:K]

def contadorClasses(listaVizinhosProximos):
  qtd_A = 0
  qtd_B = 0  
    
  for aux in listaVizinhosProximos:
    if aux[1] == 'A':
      qtd_A += 1
    elif aux[1] == 'B':
      qtd_B += 1

  return qtd_A, qtd_B

def KNN(K, X, Y, novoExemplo):
    
  distancias = getListaDistanciaOrdendada(X, Y, novoExemplo)

  vizinhosProximos = getListaKVizinhosMaisProximos(distancias, K)

  quantidadeA, quantidadeB = contadorClasses(vizinhosProximos)

  if quantidadeA > quantidadeB:
    print("A")
  else:
    print("B")

X = [
     [0, 1, 0.75], 
     [0.5, 0, 0.5],
     [1, 1, 0.75],
     [1, 0, 0],
     [0, 0, 1]
    ]

Y = ["B", "A", "B", "A", "B"]

K = 3

vetorNovo = [0.5, 0, 0.25]

KNN(K, X, Y, vetorNovo)