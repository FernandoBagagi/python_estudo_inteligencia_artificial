import numpy as np
import pytest

from classifiers.knn_classifier import DistanceMetric, KNNClassifier

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def valid_2d_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Retorna um dataset 2D simples normalizado com rótulos inteiros."""
    x_train = np.array(
        [
            [0.1, 0.1],  # Classe 0
            [0.2, 0.2],  # Classe 0
            [0.8, 0.8],  # Classe 1
            [0.9, 0.9],  # Classe 1
        ]
    )
    y_train = np.array([0, 0, 1, 1])
    return x_train, y_train


@pytest.fixture
def high_dimensional_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Retorna um dataset 5D (5 características por amostra)."""
    x_train = np.array(
        [
            [0.1, 0.2, 0.1, 0.0, 0.3],
            [0.9, 0.8, 0.7, 0.9, 1.0],
        ]
    )
    y_train = np.array([10, 20])
    return x_train, y_train


# ==============================================================================
# Testes de Inicialização e Parâmetros
# ==============================================================================


@pytest.mark.parametrize('invalid_k', [0, -1, -10])
def test_init_invalid_k_raises_error(invalid_k: int) -> None:
    """Garante erro ao tentar instanciar o KNN com k <= 0."""
    with pytest.raises(ValueError, match='maior que zero'):
        KNNClassifier(k=invalid_k)


def test_init_default_parameters() -> None:
    """Valida se a inicialização usa os valores padrão corretos."""
    knn = KNNClassifier()
    assert knn.k == 3
    assert knn.distance_metric == DistanceMetric.EUCLIDEAN


# ==============================================================================
# Testes de Validação do Método fit()
# ==============================================================================


def test_fit_row_count_mismatch(
    valid_2d_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    """Lança erro quando número de linhas de x_train e y_train difere."""
    x_train, _ = valid_2d_dataset
    y_incompatible = np.array([0, 1])
    knn = KNNClassifier(k=2)

    with pytest.raises(ValueError, match='mesmo número de linhas'):
        knn.fit(x_train, y_incompatible)


def test_fit_k_greater_than_samples(
    valid_2d_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    """Lança erro quando k é maior que a quantidade de amostras de treino."""
    x_train, y_train = valid_2d_dataset
    knn = KNNClassifier(k=10)

    with pytest.raises(ValueError, match='não pode ser maior'):
        knn.fit(x_train, y_train)


def test_fit_k_equal_to_samples(
    valid_2d_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    """Permite ajustar o modelo quando k é exatamente igual ao total de amostras."""
    x_train, y_train = valid_2d_dataset
    knn = KNNClassifier(k=4)
    fitted_model = knn.fit(x_train, y_train)

    assert fitted_model is knn


@pytest.mark.parametrize(
    'invalid_x',
    [
        np.array([[-0.1, 0.5], [0.2, 0.3]]),  # Valor < 0.0
        np.array([[0.1, 1.5], [0.2, 0.3]]),  # Valor > 1.0
        np.array([[-1.0, 2.0], [0.5, 0.5]]),  # Ambos fora do intervalo
    ],
)
def test_fit_unnormalized_x_train_raises_error(invalid_x: np.ndarray) -> None:
    """Lança erro se x_train tiver valores fora da faixa [0.0, 1.0]."""
    y_train = np.array([0, 1])
    knn = KNNClassifier(k=1)

    with pytest.raises(ValueError, match='normalizados entre 0.0 e 1.0'):
        knn.fit(invalid_x, y_train)


@pytest.mark.parametrize(
    'invalid_y',
    [
        np.array([-1, 0, 1, 2]),  # Inteiro negativo
        np.array([0.0, 1.0, 2.0, 3.0]),  # Floats (não permitidos pela função)
        np.array(['a', 'b', 'c', 'd']),  # Strings
    ],
)
def test_fit_invalid_y_label_types(
    valid_2d_dataset: tuple[np.ndarray, np.ndarray], invalid_y: np.ndarray
) -> None:
    """Lança erro se y_train não for composto por booleanos ou inteiros >= 0."""
    x_train, _ = valid_2d_dataset
    knn = KNNClassifier(k=2)

    with pytest.raises(ValueError, match='booleanos ou números inteiros'):
        knn.fit(x_train, invalid_y)


def test_fit_valid_boolean_labels() -> None:
    """Confirma que rótulos booleanos são aceitos no fit."""
    x_train = np.array([[0.1, 0.1], [0.9, 0.9]])
    y_train = np.array([True, False])
    knn = KNNClassifier(k=1)

    assert knn.fit(x_train, y_train) is knn


# ==============================================================================
# Testes de Validação do Método predict()
# ==============================================================================


def test_predict_before_fit_raises_error() -> None:
    """Garante erro de execução ao tentar prever sem antes treinar o modelo."""
    knn = KNNClassifier(k=1)
    x_test = np.array([[0.1, 0.1]])

    with pytest.raises(RuntimeError, match='treinado com fit'):
        knn.predict(x_test)


def test_predict_column_mismatch(
    valid_2d_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    """Lança erro se x_test tiver número de colunas diferente de x_train."""
    x_train, y_train = valid_2d_dataset
    knn = KNNClassifier(k=2).fit(x_train, y_train)

    x_test_3d_features = np.array([[0.1, 0.2, 0.3]])
    with pytest.raises(ValueError, match='mesmo número de colunas'):
        knn.predict(x_test_3d_features)


# ==============================================================================
# Testes de Cálculo Matemático e Classificação
# ==============================================================================


def test_predict_euclidean_distance(
    valid_2d_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    """Valida o resultado do cálculo exato da distância Euclidiana."""
    x_train, y_train = valid_2d_dataset
    knn = KNNClassifier(k=2, distance_metric=DistanceMetric.EUCLIDEAN)
    knn.fit(x_train, y_train)

    x_test = np.array(
        [
            [0.15, 0.15],  # Deve ser predito como 0
            [0.85, 0.85],  # Deve ser predito como 1
        ]
    )

    predictions = knn.predict(x_test)
    np.testing.assert_array_equal(predictions, np.array([0, 1]))


def test_predict_manhattan_distance() -> None:
    """Valida o resultado do cálculo exato da distância de Manhattan."""
    x_train = np.array(
        [
            [0.0, 0.0],  # Distância Manhattan para (0.3, 0.0) é 0.3
            [0.5, 0.0],  # Distância Manhattan para (0.3, 0.0) é 0.2 -> Mais próximo!
        ]
    )
    y_train = np.array([100, 200])

    knn = KNNClassifier(k=1, distance_metric=DistanceMetric.MANHATTAN)
    knn.fit(x_train, y_train)

    x_test = np.array([[0.3, 0.0]])
    predictions = knn.predict(x_test)

    assert predictions[0] == 200


def test_predict_high_dimensional_data(
    high_dimensional_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    """Valida a predição para dados com 5 características por amostra."""
    x_train, y_train = high_dimensional_dataset
    knn = KNNClassifier(k=1).fit(x_train, y_train)

    x_test = np.array([[0.85, 0.75, 0.65, 0.85, 0.95]])
    predictions = knn.predict(x_test)

    assert predictions[0] == 20


def test_predict_voting_tie_resolution() -> None:
    """Verifica a estabilidade de empate na moda dos k vizinhos."""
    x_train = np.array(
        [
            [0.1, 0.1],  # Distância = 0.0 (Classe 0)
            [0.1, 0.1],  # Distância = 0.0 (Classe 1)
        ]
    )
    y_train = np.array([0, 1])

    # k=2 gera empate exato entre a classe 0 e classe 1
    knn = KNNClassifier(k=2).fit(x_train, y_train)
    predictions = knn.predict(np.array([[0.1, 0.1]]))

    # Deve retornar a primeira classe encontrada no Counter sem falhar
    assert predictions[0] in (0, 1)


def test_predict_boolean_labels_output() -> None:
    """Verifica a integridade de tipo na saída para rótulos booleanos."""
    x_train = np.array([[0.1, 0.1], [0.9, 0.9]])
    y_train = np.array([True, False])

    knn = KNNClassifier(k=1).fit(x_train, y_train)
    predictions = knn.predict(np.array([[0.15, 0.15]]))

    assert predictions[0]
