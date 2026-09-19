## Using Keras to Build and Train Neural Networks
# In this exercise we will use a neural network to predict diabetes using
# the Pima Diabetes Dataset.
# We will use the Keras package to quickly build and train a neural network.

## UCI Pima Diabetes Dataset
# UCI ML Repositiory (https://www.kaggle.com/uciml/pima-indians-diabetes-database)


## Attributes: (all numeric-valued)
# 1. Number of times pregnant
# 2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# 3. Diastolic blood pressure (mm Hg)
# 4. Triceps skin fold thickness (mm)
# 5. 2-Hour serum insulin (mu U/ml)
# 6. Body mass index (weight in kg/(height in m)^2)
# 7. Diabetes pedigree function
# 8. Age (years)
# 9. Class variable (0 or 1)
# The UCI Pima Diabetes Dataset which has 8 numerical predictors and a binary outcome.


import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Dense, Input
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print('\n\nCarregando dataset ...\n')

path = 'assets/pima-indians-diabetes.data.csv'
diabetes_df = pd.read_csv(path)

print(diabetes_df.head())

print(
    f'\nO dataset possui {diabetes_df.shape[0]} linhas e {diabetes_df.shape[1]} colunas'
)


print('\n\nSeparando colunas de entrada da coluna de saída ...')
X = diabetes_df.iloc[:, :-1]  # Pega todas as colunas, exceto a última
y = diabetes_df.iloc[:, -1]  # Pega somente a última coluna

print('\n\nSeparando 75% dos dados para treino e 25% para teste ...')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=11111
)

print('\n\nNormalizando os dados usando StandardScaler ...')
scaler = StandardScaler()
# Ajusta o scaler nos dados de treino e os transforma
X_train_norm = scaler.fit_transform(X_train)
# Transforma os dados de teste usando o scaler já ajustado aos de treino
X_test_norm = scaler.transform(X_test)

## Build a Single Hidden Layer Neural Network
# We will use the Sequential model to quickly build a neural network.
# Our first network will be a single layer network.  We have 8 variables, so we set
# the input shape to 8.  Let's start by having a single hidden layer with 12 nodes.

print('\n\nDefinindo o modelo ...')
model_1 = Sequential(
    [
        # Entrada de dimensão 8
        Input(shape=(8,)),
        # Camada oculta: 12 nós, ativação sigmoid
        Dense(12, activation='sigmoid'),
        # Camada de saída: 1 nó, ativação sigmoid
        # Padrão para classificação binária
        Dense(1, activation='sigmoid'),
    ]
)
print(model_1.summary())

print('\n\nCompilando o modelo ...')
model_1.compile(
    optimizer=SGD(learning_rate=0.003),  # Optimizer
    loss='binary_crossentropy',  # Loss Function
    metrics=['accuracy'],  # Metrics
)

input('Pressione ENTER para continuar o script...')

print('\n\nTreinando o modelo ...')
run_hist_1 = model_1.fit(
    X_train_norm, y_train, validation_data=(X_test_norm, y_test), epochs=200
)
# the fit function returns the run history.
# It is very convenient, as it contains information about the model fit, iterations etc.

# Faça a predição do modelo nos dados de teste "X_test_norm"
# use a variável y_pred_prob_nn_1 para armazenar as probalidades
# (saída da função predict)
# Faz a predição das probabilidades
y_pred_prob_nn_1 = model_1.predict(X_test_norm)

# use a variável y_pred_class_nn_1 para armazenar a resposta binária, ou seja, 1 se
# a probabilidade for maior do que .5 ou 0 caso contrário
# Converte as probabilidades em classes binárias (1 se > 0.5, caso contrário 0)
y_pred_class_nn_1 = (y_pred_prob_nn_1 > 0.5).astype(int)

print('\n\nPredições')
print('Probabilidade -> Predição')
for i in range(10):
    print(f'{y_pred_prob_nn_1[i][0]:.11f} ->     {y_pred_class_nn_1[i][0]}')

print(f'\naccuracy is {accuracy_score(y_test, y_pred_class_nn_1):.4f}')

# Let's look at the `run_hist_1` object that was created, specifically
# its `history` attribute.

print('\n\nAs chaves são:')
for key in run_hist_1.history:
    print('  ', key)

# Let's plot the training loss and the validation loss over the different epochs and
# see how it looks.

fig, ax = plt.subplots()
ax.plot(run_hist_1.history['loss'], 'r', marker='.', label='Train Loss')
ax.plot(run_hist_1.history['val_loss'], 'b', marker='.', label='Validation Loss')
ax.legend()

plt.show()

# Looks like the losses are still going down on both the training set and
# the validation set.  This suggests that the model might benefit from further training.
# Let's train the model a little more and see what happens. Note that it will pick up
# from where it left off. Train for 1000 more epochs."""

## Note that when we call "fit" again, it picks up where it left off
print('\n\nTreinando o modelo mais um pouco ...')
run_hist_1b = model_1.fit(
    X_train_norm, y_train, validation_data=(X_test_norm, y_test), epochs=1000
)

n = len(run_hist_1.history['loss'])
m = len(run_hist_1b.history['loss'])
fig, ax = plt.subplots(figsize=(16, 8))

ax.plot(
    range(n), run_hist_1.history['loss'], 'r', marker='.', label='Train Loss - Run 1'
)
ax.plot(
    range(n, n + m),
    run_hist_1b.history['loss'],
    'hotpink',
    marker='.',
    label='Train Loss - Run 2',
)

ax.plot(
    range(n),
    run_hist_1.history['val_loss'],
    'b',
    marker='.',
    label='Validation Loss - Run 1',
)
ax.plot(
    range(n, n + m),
    run_hist_1b.history['val_loss'],
    'LightSkyBlue',
    marker='.',
    label='Validation Loss - Run 2',
)

ax.legend()

plt.show()

# Note that this graph begins where the other left off.  While the training loss is
# still going down, it looks like the validation loss has stabilized
# (or even gotten worse!).  This suggests that our network will not benefit from further
# training.  What is the appropriate number of epochs?
