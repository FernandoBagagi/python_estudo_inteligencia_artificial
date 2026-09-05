import numpy as np


class Ponto3D:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


class DistanciaVizinho:
    def __init__(self, distancia: float, label_vizinho: str):
        self.distancia = distancia
        self.label_vizinho = label_vizinho


class KnnAulaIA:
    label_classe_a = 'A'
    label_classe_b = 'B'

    def __init__(self, k: int):
        self.k = k

    def fit(self, x: list[Ponto3D], y: list[str]) -> None:
        self.x = x
        self.y = y

    def _calcular_distacia_euclidiana(self, a: Ponto3D, b: Ponto3D) -> float:
        # A distância euclidiana entre dois pontos é calculada pela fórmula:
        # d(a,b) = ( (xa-xb)^2 + (ya-yb)^2 + (za-zb)^2 )^(1/2)
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** (1 / 2)

    def _gerar_lista_distancia_vizinho(
        self, novo_exemplo: Ponto3D
    ) -> list[DistanciaVizinho]:

        lista_distancia_vizinho: list[DistanciaVizinho] = []

        for i in range(len(self.x)):
            distancia = self._calcular_distacia_euclidiana(self.x[i], novo_exemplo)
            label_vizinho = self.y[i]

            lista_distancia_vizinho.append(DistanciaVizinho(distancia, label_vizinho))

        return lista_distancia_vizinho

    def _ordernar_por_distancia(
        self,
        lista_distancia_vizinho: list[DistanciaVizinho],
    ) -> None:
        lista_distancia_vizinho.sort(key=lambda elemento: elemento.distancia)

    def _get_k_vizinhos_mais_proximos(
        self, lista_distancia_vizinho: list[DistanciaVizinho]
    ) -> list[DistanciaVizinho]:
        return lista_distancia_vizinho[: self.k]

    def _predizer_classe(self, vizinhos_mais_proximos: list[DistanciaVizinho]) -> str:

        qtde_a = 0
        qtde_b = 0

        for vizinho in vizinhos_mais_proximos:
            label = vizinho.label_vizinho
            if label == KnnAulaIA.label_classe_a:
                qtde_a += 1
            elif label == KnnAulaIA.label_classe_b:
                qtde_b += 1

        if qtde_a == qtde_b:
            classes = (KnnAulaIA.label_classe_a, KnnAulaIA.label_classe_b)
            return np.random.default_rng(seed=42).choice(classes)
        elif qtde_a > qtde_b:
            return KnnAulaIA.label_classe_a
        else:
            return KnnAulaIA.label_classe_b

    def predict_one(self, ponto: Ponto3D) -> str:
        distacias = self._gerar_lista_distancia_vizinho(ponto)
        self._ordernar_por_distancia(distacias)
        vizinhos_mais_proximos = self._get_k_vizinhos_mais_proximos(distacias)
        return self._predizer_classe(vizinhos_mais_proximos)

    def predict_all(self, pontos: list[Ponto3D]) -> list[str]:

        return [self.predict_one(ponto) for ponto in pontos]


def main() -> None:

    x = [
        Ponto3D(0.0, 1.0, 0.75),
        Ponto3D(0.5, 0.0, 0.50),
        Ponto3D(1.0, 1.0, 0.75),
        Ponto3D(1.0, 0.0, 0.00),
        Ponto3D(0.0, 0.0, 1.00),
    ]

    y = [
        KnnAulaIA.label_classe_b,
        KnnAulaIA.label_classe_a,
        KnnAulaIA.label_classe_b,
        KnnAulaIA.label_classe_a,
        KnnAulaIA.label_classe_b,
    ]

    knn = KnnAulaIA(k=3)

    knn.fit(x, y)

    x_predict = Ponto3D(0.5, 0.0, 0.25)

    y_predict = knn.predict_one(x_predict)

    print(y_predict)


if __name__ == '__main__':
    main()
