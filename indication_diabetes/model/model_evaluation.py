from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.callbacks import EarlyStopping

# load model 
model = tf.keras.models.load_model('/home/brunopaiva/DataSet/Indicador_diabetes/indication_diabetes/model/model.keras')

# load data
saida_test = pd.read_csv('/home/brunopaiva/DataSet/Indicador_diabetes/indication_diabetes/dataset/test/output_test.csv')
input_test = pd.read_csv('/home/brunopaiva/DataSet/Indicador_diabetes/indication_diabetes/dataset/test/input_test_standard.csv')

#Carregando o custo
history = pd.read_csv('/home/brunopaiva/DataSet/Indicador_diabetes/indication_diabetes/model/custo.csv')

# Plot training & validation accuracy values
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.yscale("log")
plt.title('Função Perda')
plt.xlabel('Epocas')
plt.legend(['Treino', 'Teste'], loc='upper left')
plt.grid()
plt.show()

#Previsões no conjunto de teste
output_model_ = model.predict(input_test)
y_test_class = saida_test.values

output_model_ = (output_model_ >= 0.5).astype(int)

print(output_model_)
print("y_test:",y_test_class)
print(saida_test.shape)
print(output_model_.shape)

for i in range(len(output_model_)):
    if(output_model_[i]>= 0.5):
      output_model_[i] = 1
    else:
      output_model_[i] = 0
print(output_model_)

saida_test = saida_test.values.reshape(-1, 1)  # Ajustar para ser uma coluna
output_model_ = output_model_.reshape(-1, 1)  # Ajustar para ser uma coluna

# precision = precision_score(y_true, y_pred, zero_division=1)

print('Acurácia:', accuracy_score(y_test_class, output_model_))
print('Precisão:', precision_score(y_test_class, output_model_, zero_division= 1.0))
print('Sensibilidade:', recall_score(y_test_class, output_model_))
print('F1-Score:', f1_score(y_test_class, output_model_))

cm = confusion_matrix(y_test_class, output_model_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
disp.ax_.set_title('Matriz de confusão')
disp.ax_.set_xlabel('Classificação prevista')
disp.ax_.set_ylabel('Classificação real')
plt.show()