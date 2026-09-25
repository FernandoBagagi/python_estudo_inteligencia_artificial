import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Input
from keras.models import Sequential
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

# Let's explore the dataset a little bit

# Load the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print('Shape data:', x_train[0].shape)

# Let's just look at a particular example to see what is inside

print('x_train[333]', x_train[333])  # Just a 28 x 28 numpy array of ints from 0 to 255

# What is the corresponding label in the training set?
print('y_train[333]', y_train[333])

# Let's see what this image actually looks like

plt.imshow(x_train[333], cmap='Greys_r')

plt.show()

# this is the shape of the np.array x_train
# it is 3 dimensional.
print(x_train.shape, 'train samples')
print(x_test.shape, 'test samples')

## For our purposes, these images are just a vector of 784 inputs, so let's convert
x_train = x_train.reshape(len(x_train), 28 * 28)
x_test = x_test.reshape(len(x_test), 28 * 28)

## Keras works with floats, so we must cast the numbers to floats
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

## Normalize the inputs so they are between 0 and 1
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print('y_train[333]', y_train[333])
# now the digit k is represented by a 1 in the kth entry (0-indexed) of 
# the length 10 vector

# We will build a model with one hidden layer of size 64
# Fully connected inputs at each layer
# We will use dropout of .2 to help regularize
model_1 = Sequential(
    [
        Input(shape=(784,)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax'),
    ]
)

## Note that this model has a LOT of parameters
model_1.summary()

# Let's compile the model
learning_rate = 0.001
model_1.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(learning_rate=learning_rate),
    metrics=['accuracy'],
)
# note that `categorical cross entropy` is the natural generalization
# of the loss function we had in binary classification case, to multi class case

# And now let's fit.

batch_size = 128  # mini-batch with 128 examples
epochs = 30
history = model_1.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose='1',
    validation_data=(x_test, y_test),
)

## We will use Keras evaluate function to evaluate performance on the test set

score = model_1.evaluate(x_test, y_test, verbose='0')
print('Test loss:', score[0])
print('Test accuracy:', score[1])


def plot_loss_accuracy(history):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(history.history['loss'], 'r-x', label='Train Loss')
    ax.plot(history.history['val_loss'], 'b-x', label='Validation Loss')
    ax.legend()
    ax.set_title('cross_entropy loss')
    ax.grid(True)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(history.history['accuracy'], 'r-x', label='Train Accuracy')
    ax.plot(history.history['val_accuracy'], 'b-x', label='Validation Accuracy')
    ax.legend()
    ax.set_title('accuracy')
    ax.grid(True)
    plt.show()


plot_loss_accuracy(history)

# This is reasonably good performance, but we can do even better!
# Next you will build an even bigger network and compare the performance.

# Exercise
# Your Turn: Build your own model
# Use the Keras "Sequential" functionality to build `model_2` with the following
# specifications:

# 1. Two hidden layers.
# 2. First hidden layer of size 400 and second of size 300
# 3. Dropout of .4 at each layer
# 4. How many parameters does your model have?  How does it compare with the previous
# model?
# 4. Train this model for 20 epochs with RMSProp at a learning rate of .001 and a batch
# size of 128

# SOLUTION


model_2 = Sequential(
    [
        Input(shape=(784,)),
        Dense(400, activation='relu'),
        Dropout(0.4),
        Dense(300, activation='relu'),
        Dropout(0.4),
        Dense(10, activation='softmax'),
    ]
)

model_2.summary()

model_2.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(learning_rate=0.001),
    metrics=['accuracy'],
)

early_stopping = EarlyStopping(
    monitor='val_loss',  # Métrica que será monitorada
    patience=3,  # Número de épocas sem melhoria até interromper
    verbose=1,  # Exibe mensagem no console quando parar
    restore_best_weights=True,
    # Restaura os melhores pesos encontrados, não os da última época
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

history = model_2.fit(
    x_train_sub,
    y_train_sub,
    batch_size=128,
    epochs=20,
    verbose='1',
    validation_data=(x_val, y_val),
    callbacks=[early_stopping],
)

score = model_2.evaluate(x_test, y_test, verbose='0')
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
