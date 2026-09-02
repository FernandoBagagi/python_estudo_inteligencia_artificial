import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from classifiers.knn_classifier import DistanceMetric, KNNClassifier

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

knn = KNNClassifier(k=3)

# Treinamento

knn = knn.fit(X_train.to_numpy(), y_train.to_numpy())

# Predição

y_predict = knn.predict(X_test.to_numpy())

# Matriz de Confusão

ConfusionMatrixDisplay.from_predictions(y_test, y_predict, cmap=plt.cm.Blues)
plt.title('Matriz de Confusão')
plt.show()

# Acurácia

accuracy = accuracy_score(y_test, y_predict)
print(f'Acurácia: {accuracy:.4f}')
