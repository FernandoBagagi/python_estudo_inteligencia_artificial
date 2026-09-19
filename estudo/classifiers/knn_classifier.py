from collections import Counter
from enum import StrEnum

import numpy as np


class DistanceMetric(StrEnum):
    EUCLIDEAN = 'euclidean'
    MANHATTAN = 'manhattan'


class KNNClassifier:
    def __init__(
        self,
        k: int = 3,
        distance_metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
    ):
        if k <= 0:
            raise ValueError('O parâmetro k deve ser um inteiro maior que zero!')

        self.k = k
        self.distance_metric = distance_metric

    def _is_y_train_type_valid(self, y_train: np.ndarray) -> bool:

        # Checa se é booleano
        if np.issubdtype(y_train.dtype, np.bool_):
            return True

        # Checa se é inteiro e se o menor valor é >= 0
        if np.issubdtype(y_train.dtype, np.integer):
            return bool(np.min(y_train) >= 0)

        return False

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
    ) -> 'KNNClassifier':

        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                'O conjunto de treinamento (x_train e y_train) '
                'devem ter o mesmo número de linhas!'
            )

        if self.k > x_train.shape[0]:
            raise ValueError(
                'O valor de k não pode ser maior que o número de amostras no treino!'
            )

        min_x_train_val = np.min(x_train)
        if min_x_train_val < 0.0  and not np.isclose(min_x_train_val, 0.0):
            print (f'Foi encontrado o valor {min_x_train_val}')
            raise ValueError(
                'Todos os valores de x_train devem estar normalizados maior 0.0!'
            )

        max_x_train_val = np.max(x_train)
        if  max_x_train_val > 1.0 and not np.isclose(max_x_train_val, 1.0):
            print (f'Foi encontrado o valor {max_x_train_val}')
            raise ValueError(
                'Todos os valores de x_train devem estar normalizados menor 1.0!'
            )

        if not self._is_y_train_type_valid(y_train):
            raise ValueError(
                'Os rótulos (y_train) devem conter apenas valores booleanos '
                'ou números inteiros maiores/igual a zero'
            )

        self.x_train = x_train
        self.y_train = y_train

        return self

    def _squared_differences(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Calcula a Diferença Quadrática entre dois vetores.

        Subtrai elemento a elemento e eleva ao quadrado a diferença.
        \n(a - b) ^ 2

        Parameters
        ----------
        a : np.ndarray
            Vetor de características do primeiro ponto.
        b : np.ndarray
            Vetor de características do segundo ponto.

        Returns
        -------
        np.ndarray
            A diferença quadrática entre cada elemento dos vetores 'a' e 'b'.
        """

        return (a - b) ** 2

    def _euclidean_distance(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Calcula a Distância Euclidiana entre dois vetores.

        A distância é calculada pela fórmula:
        d(a, b) = sqrt( sum( (a_i - b_i)^2 ) )

        Parameters
        ----------
        a : np.ndarray
            Vetor de características do primeiro ponto.
        b : np.ndarray
            Vetor de características do segundo ponto.

        Returns
        -------
        np.ndarray
            A distância em linha reta entre os pontos 'a' e 'b'.
        """

        squared_differences = self._squared_differences(a, b)

        sum_of_squares = np.sum(squared_differences, axis=1)

        return np.sqrt(sum_of_squares)

    def _absolute_differences(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Calcula a diferença absoluta entre dois vetores.

        Subtrai elemento a elemento e faz modulo da diferença.
        \n|a - b|

        Parameters
        ----------
        a : np.ndarray
            Vetor de características do primeiro ponto.
        b : np.ndarray
            Vetor de características do segundo ponto.

        Returns
        -------
        np.ndarray
            A diferença absoluta entre cada elemento dos vetores 'a' e 'b'.
        """

        return np.abs(a - b)

    def _manhattan_distance(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Calcula a Distância de Manhattan entre dois vetores.

        A distância é calculada pela fórmula:
        d(a, b) = sum( |a_i - b_i| )

        Parameters
        ----------
        a : np.ndarray
            Vetor de características do primeiro ponto.
        b : np.ndarray
            Vetor de características do segundo ponto.

        Returns
        -------
        np.ndarray
            A distância em grade entre os pontos 'a' e 'b'.
        """

        absolute_differences = self._absolute_differences(a, b)

        return np.sum(absolute_differences, axis=1)

    def _calculate_distance(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.distance_metric == DistanceMetric.EUCLIDEAN:
            return self._euclidean_distance(a, b)
        else:
            return self._manhattan_distance(a, b)

    def _predict_one(self, x_test_row: np.ndarray):

        # Calcula as distâncias entre todo o x_train e a linha de teste
        distances = self._calculate_distance(self.x_train, x_test_row)

        # Ordena pela distância e pega os indices dos k vizinhos
        k_neighbors_indexes = np.argsort(distances)[: self.k]

        # Pega as classes dos k vizinhos
        k_neighbors_classes = self.y_train[k_neighbors_indexes]

        # Faz a moda das classes
        return Counter(k_neighbors_classes).most_common(1)[0][0]

    def predict(self, x_test: np.ndarray) -> np.ndarray:

        if not hasattr(self, 'x_train'):
            raise RuntimeError(
                'O modelo deve ser treinado com fit() antes de realizar predições!'
            )

        if x_test.shape[1] != self.x_train.shape[1]:
            raise ValueError(
                'O teste deve ter o mesmo número de colunas que treinamento!'
            )

        return np.array([self._predict_one(row) for row in x_test])
