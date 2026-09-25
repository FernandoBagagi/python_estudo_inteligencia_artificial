import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.layers import Dense, Dropout, Input
from keras.models import Sequential
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print('Shape data: ', x_train[217].shape)

print('x_train[217]:\n', x_train[217])

print('y_train[217]:', y_train[217])

print('Shape train samples: ', x_train.shape)
print('Shape test samples: ', x_test.shape)
x_train = x_train.reshape(len(x_train), 32 * 32 * 3)
x_test = x_test.reshape(len(x_test), 32 * 32 * 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

print(x_train[217].dtype)

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print('y_train[217]', y_train[217])

model = Sequential(
    [
        Input(shape=(32 * 32 * 3,)),
        Dense(400, activation='relu'),
        Dropout(0.4),
        Dense(300, activation='relu'),
        Dropout(0.4),
        Dense(10, activation='softmax'),
    ]
)

model.summary()

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

# Converte y_train para um array NumPy garantido
y_array = np.asarray(y_train)

x_train_sub, x_val, y_train_sub, y_val = train_test_split(
    x_train,
    y_train,
    test_size=1 / 3,
    random_state=42,
    stratify=y_array.argmax(axis=1)
    if y_array.ndim > 1
    else y_array,  # garante proporção das classes
)

history = model.fit(
    x_train_sub,
    y_train_sub,
    batch_size=128,
    epochs=20,
    verbose='1',
    validation_data=(x_val, y_val),
    callbacks=[early_stopping],
)

score = model.evaluate(x_test, y_test, verbose='0')
print('Test loss:', score[0])
print('Test accuracy:', score[1])


def plot_loss_accuracy2(history):
    fig = plt.figure(figsize=(14, 6))

    # Épocas executadas
    epochs = range(1, len(history.history['loss']) + 1)

    # Best epoch index (com base no menor val_loss)
    best_epoch = float(np.argmin(history.history['val_loss'])) + 1

    # 1. Gráfico de Loss
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(epochs, history.history['loss'], 'r-x', label='Train Loss')
    ax1.plot(epochs, history.history['val_loss'], 'b-x', label='Validation Loss')

    # Marcar o ponto de melhor val_loss (onde o restore_best_weights salva)
    ax1.axvline(
        x=best_epoch, color='g', linestyle='--', label=f'Melhor Época ({best_epoch})'
    )

    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Cross Entropy Loss')
    ax1.grid(True)

    # 2. Gráfico de Acurácia
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
    ax2.set_title('Accuracy')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# Chamada da função
plot_loss_accuracy2(history)
