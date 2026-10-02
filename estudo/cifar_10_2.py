import time

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from keras.models import Sequential
from keras.optimizers import RMSprop
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Carrega os dados
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


keras.utils.set_random_seed(42)


# ===================================================================================
# CONCEITO: One-Hot Encoding
# ===================================================================================
# Transforma os rótulos numéricos em vetores binários de tamanho igual ao número de
# classes. Exemplo, em um conjunto de 10 classes, transforma a classe 5 em
# [0, 0, 0, 0, 0, 1, 0, 0, 0, 0].
#
# VANTAGENS
#
# Evita Falsa Ordem:
#   Se usar apenas números (0 a 9), a rede deduziria que a classe 9 (Caminhão) tem
#   "mais valor" ou está mais longe da 1 (Carro) do que da 3 (Gato).
#
# Compatibilidade com Softmax:
#   A função Softmax força a soma das saídas a ser 1.0 (ou 100%), transformando os
#   valores da rede em probabilidades para cada classe. Exemplo, usando a softmax
#   a saída do modelo [0.01, 0.01, 0.01, 0.01, 0.02, 0.89, 0.02, 0.01, 0.01, 0.01]
#   indica que há 89% de chance de ser a classe 5 e não um valor (ex: 4,87). Assim
#   a função de perda (Cross-Entropy) compara essa saída com o vetor Alvo (One-Hot)
#   [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00] e ajusta os pesos
# ===================================================================================

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# ===================================================================================
# CONCEITO: Normalização dos dados
# ===================================================================================
#
# .astype('float32'):
#   Converte os pixels de uint8 (0 a 255) para o tipo float32.
#   O float32 é o tipo padrão usado em processamento rápido em GPUs e no Keras.
#
# / 255.0:
#   Normaliza os valores de pixel, que estão no intervalo de 0 a 255, para o
#   intervalo 0.0, 1.0.
# ===================================================================================

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# ===================================================================================
# CONCEITO: Arquitetura da Rede Neural Densa (MLP - Multi-Layer Perceptron)
# ===================================================================================
# Input(shape=(32, 32, 3)):
#   Define o formato de entrada para (32, 32, 3) pois o CIFAR-10 possui imagens de
#   32 pixels de largura, 32 de altura e 3 canais de cor (R, G, B).
#
# Flatten:
#   Achata a matriz 3D da imagem (32, 32, 3) em um único vetor unidimensional de
#   3.072 posições (32 * 32 * 3). Isso torna o modelo autossuficiente pois a rede
#   aceita a imagem no seu formato natural e faz a transformação internamente.
#
# Dense com activation='relu':
#   Camada de neurônios totalmente conectada. A função de ativação 'relu'
#   (Rectified Linear Unit) transforma valores negativos em 0. Isso introduz
#   não-linearidade o que permite a rede aprender padrões complexos. É um padrão usar
#   potências de 2 (64, 128, 256, ...) porque otimiza o alocamento de memória das GPUs.
#
# Dropout:
#   Técnica de regularização que desliga aleatoriamente uma porcentagem dos neurônios
#   da camada anterior a cada passo de treinamento. Isso força a rede a não especializar
#   demais alguns neurônios e perder generalização (overfitting). Os valores ideais
#   ficam entre 0.2 a 0.5 (20% a 50%), se for pouco demais não faz efeito se for muito
#   a rede perde capacidade de aprender.
#
# Dense com activation='softmax':
#   Camada de saída com 10 neurônios (um para cada classe do CIFAR-10). A ativação
#   'softmax' converte os valores finais em uma distribuição de probabilidades que
#   somam 1.0 (100%).
# ===================================================================================

model = Sequential(
    [
        Input(shape=(32, 32, 3)),
        Flatten(),
        Dense(2048, activation='relu'),
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(10, activation='softmax'),
    ]
)

model_cnn = Sequential(
    [
        Input(shape=(32, 32, 3)),
        # Bloco 1
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),  # Reduz de 32x32 para 16x16
        Dropout(0.2),
        # Bloco 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),  # Reduz de 16x16 para 8x8
        Dropout(0.3),
        # Bloco 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),  # Reduz de 8x8 para 4x4
        Dropout(0.4),
        # Classificador
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax'),
    ]
)

model.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(learning_rate=0.001),
    metrics=['accuracy'],
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True,
)

x_train_sub, x_val, y_train_sub, y_val = train_test_split(
    x_train,
    y_train,
    test_size=1 / 3,
    random_state=42,
    stratify=np.asarray(y_train).argmax(axis=1),
)

print('Iniciando o treinamento do modelo...')

inicio_treino = time.perf_counter()

# 4. Treinamento
history = model.fit(
    x_train_sub,
    y_train_sub,
    batch_size=128,
    epochs=30,
    verbose='1',
    validation_data=(x_val, y_val),
    callbacks=[early_stopping],
)

fim_treino = time.perf_counter()


# 5. Métricas e Gráficos
def plot_loss_accuracy(history):
    fig = plt.figure(figsize=(14, 6))
    epochs = range(1, len(history.history['loss']) + 1)
    best_epoch = float(np.argmin(history.history['val_loss'])) + 1

    # Loss
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(epochs, history.history['loss'], 'r-x', label='Train Loss')
    ax1.plot(epochs, history.history['val_loss'], 'b-x', label='Validation Loss')
    ax1.axvline(
        x=best_epoch, color='g', linestyle='--', label=f'Melhor Época ({best_epoch})'
    )
    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Cross Entropy Loss - CNN')
    ax1.grid(True)

    # Accuracy
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(epochs, history.history['accuracy'], 'r-x', label='Train Accuracy')
    ax2.plot(
        epochs, history.history['val_accuracy'], 'b-x', label='Validation Accuracy'
    )
    ax2.axvline(
        x=best_epoch, color='g', linestyle='--', label=f'Melhor Época ({best_epoch})'
    )
    ax2.set_xlabel('Épocas')
    ax2.set_ylabel('Acurácia')
    ax2.legend()
    ax2.set_title('Accuracy - CNN')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


plot_loss_accuracy(history)

# Conversões para Métricas do Sklearn
y_probs = model.predict(x_test)
y_probs = np.asarray(y_probs, dtype=np.float64)
y_test_np = np.asarray(y_test, dtype=np.float64)

y_pred_classes = np.argmax(y_probs, axis=1)
y_true_classes = np.argmax(y_test_np, axis=1)

cifar10_labels = [
    'Avião',
    'Carro',
    'Pássaro',
    'Gato',
    'Cervo',
    'Cão',
    'Sapo',
    'Cavalo',
    'Navio',
    'Caminhão',
]

# Matriz de Confusão
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(
    y_true_classes,
    y_pred_classes,
    display_labels=cifar10_labels,
    cmap=plt.cm.Blues,
    xticks_rotation=45,
    ax=ax,
)
plt.title('Matriz de Confusão - CNN CIFAR-10')
plt.tight_layout()
plt.show()

# Imprimir tempo
tempo_treino_segundos = fim_treino - inicio_treino
minutos = int(tempo_treino_segundos // 60)
segundos = tempo_treino_segundos % 60

print(
    '\n⏱️  Tempo total de Treinamento: ',
    f'{minutos}m {segundos:.2f}s ({tempo_treino_segundos:.2f}',
    ' segundos)',
)

# Imprimir Resultados
acc = accuracy_score(y_true_classes, y_pred_classes)
roc_auc = roc_auc_score(y_test_np, y_probs, multi_class='ovr')

print(f'\n[CNN] Acurácia no Teste: {acc * 100:.2f}%')
print(f'[CNN] Área sob a Curva ROC (AUC - OvR): {roc_auc:.4f}')
