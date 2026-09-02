import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

# Importar dados

filepath = 'assets/dataset_baseball.arff'
data, meta = arff.loadarff(filepath)

data = pd.DataFrame(data)

# Tratar e separar dados

data = data.drop(columns=['Player'])

X_data = data.iloc[:, :-1]
X_data = pd.get_dummies(X_data, drop_first=True, dtype=int)

y_data = (
    data.iloc[:, -1]
    .apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    .astype(int)
)

print('y_data')
print(y_data.dtypes)
print(y_data.head(5).T)

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.3, random_state=42, stratify=y_data
)

# Substituir NaN pela mediana

imputer = SimpleImputer(strategy='median')

X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

# Normalizar dados

scaler = MinMaxScaler()

X_train_norm = scaler.fit_transform(X_train_imp)
X_test_norm = scaler.transform(X_test_imp)

X_train = pd.DataFrame(X_train_norm, columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(X_test_norm, columns=X_test.columns, index=X_test.index)

print('X_train')
print(X_train.dtypes)
print(X_train.head(5).T)

# Construção do modelo 📈

knn = KNeighborsClassifier(n_neighbors=9, weights='distance')

# Treinamento

knn = knn.fit(X_train, y_train)

# Predição

y_predict = knn.predict(X_test)

# Matriz de Confusão

ConfusionMatrixDisplay.from_predictions(y_test, y_predict, cmap=plt.cm.Blues)
plt.title('Matriz de Confusão')
plt.show()

# Acurácia

accuracy = accuracy_score(y_test, y_predict)
print(f'Acurácia: {accuracy:.4f}')

# Área Sob a Curva ROC (ROC AUC)

# Para a curva ROC é necessário a probabilidade da classe positiva geradas
# por predict_proba, e não as previsões brutas (y_predict), garantindo a
# medição exata do desempenho da taxa de verdadeiros positivos contra falsos
# positivos em diferentes limiares.
y_probs = knn.predict_proba(X_test)

roc_auc = roc_auc_score(y_test, y_probs, multi_class='ovr')
print(f'Área sob a Curva ROC (AUC - OvR): {roc_auc:.4f}')

# Curva ROC

plt.figure(figsize=(8, 6))

# 1. Plotar a curva ROC para cada uma das 3 classes
for i in range(3):
    # Cria um vetor binário: 1 se for a classe i, 0 se for outra classe
    y_true_binary = y_test == i

    # Calcula a curva para a classe i usando a probabilidade da coluna i
    fpr, tpr, _ = roc_curve(y_true_binary, np.asarray(y_probs)[:, i])
    class_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'Classe {i} (AUC = {class_auc:.2f})')

# 2. Desenhar a linha diagonal de referência (Classificador Aleatório)
plt.plot([0, 1], [0, 1], 'k--', label='Aleatório')

plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.title('Curva ROC Multiclasse (One-vs-Rest)')
plt.legend()
plt.show()
