import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

# Importar dados

filepath = 'assets/Orange_Telecom_Churn_Data.csv'
data = pd.read_csv(filepath)

# Tratar e separar dados

data = data.drop(columns=['phone_number', 'area_code', 'state'])

X_data = data.iloc[:, :-1]
X_data = pd.get_dummies(X_data, drop_first=True, dtype=int)

y_data = data.iloc[:, -1]  # Mesma coisa de data['churned']


X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.3, random_state=42, stratify=y_data
)
# O stratify garante que a proporção de classes seja exatamente a mesma tanto no
# conjunto de treino (y_train) quanto no de teste (y_test).

# Normalizar dados

scaler = MinMaxScaler()

X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

X_train = pd.DataFrame(X_train_norm, columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(X_test_norm, columns=X_test.columns, index=X_test.index)

print('X_train')
print(X_train.dtypes)
print(X_train.head(5).T)

# Construção do modelo

knn = KNeighborsClassifier(n_neighbors=3)

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
y_probs = np.asarray(knn.predict_proba(X_test))[:, 1]

roc_auc = roc_auc_score(y_test, y_probs)
print(f'Área sob a Curva ROC (AUC): {roc_auc:.4f}')

# Curva ROC

RocCurveDisplay.from_predictions(y_test, y_probs)
plt.title('Curva ROC')
plt.plot([0, 1], [0, 1], 'k--', label='Classificador Aleatório')  # Linha de referência
plt.legend()
plt.show()
