from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# load model 
model = tf.keras.models.load_model('/home/brunopaiva/DataSet/Indicador_diabetes/indication_diabetes/model/model.keras')

# load data
saida_test = pd.read_csv('/home/brunopaiva/DataSet/Indicador_diabetes/indication_diabetes/dataset/test/output_test.csv')
x_test_padrao = pd.read_csv('/home/brunopaiva/DataSet/Indicador_diabetes/indication_diabetes/dataset/train/input_train_standard.csv')

#Ajuste para o número correto de features
x_test_padrao = x_test_padrao.iloc[:, :21] 

#Carregando o custo
history = pd.read_csv('/home/brunopaiva/DataSet/Indicador_diabetes/indication_diabetes/model/custo.csv')

# Plot training & validation accuracy values
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.yscale("log")
plt.title('Função Perda')
plt.ylabel('MSE')
plt.xlabel('Epocas')
plt.legend(['Treino', 'Teste'], loc='upper left')
plt.grid()
plt.show()

#Previsões no conjunto de teste
output_model_ = model.predict(x_test_padrao)
# y_test_class = saida_test.values

output_model_ = (output_model_ >= 0.5).astype(int)

#Métricas de avaliação
mse = mean_squared_error(saida_test, x_test_padrao)
mae = mean_absolute_error(saida_test, x_test_padrao)
r2 = r2_score(saida_test, x_test_padrao)

for i in range(len(output_model_)):
    if(output_model_[i]>= 0.5):
      output_model_[i] = 1
    else:
      output_model_[i] = 0
print(output_model_)


print('Acurácia:', accuracy_score(saida_test, output_model_))
print('Precisão:', precision_score(saida_test, output_model_))
print('Sensibilidade:', recall_score(saida_test, output_model_))
print('F1-Score:', f1_score(saida_test, output_model_))


cm = confusion_matrix(saida_test, output_model_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
disp.ax_.set_title('Matriz de confusão')
disp.ax_.set_xlabel('Classificação prevista')
disp.ax_.set_ylabel('Classificação real')
plt.show()