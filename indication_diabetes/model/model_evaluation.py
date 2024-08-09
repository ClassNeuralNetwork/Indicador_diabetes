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
from tensorflow.keras.callbacks import EarlyStopping

# load model 
model = tf.keras.models.load_model('/home/brunopaiva/DataSet/Indicador_diabetes/indication_diabetes/model/model.keras')

# load data
history = pd.read_csv('/home/brunopaiva/DataSet/Indicador_diabetes/indication_diabetes/model/custo.csv')
saida_test = pd.read_csv('/home/brunopaiva/DataSet/Indicador_diabetes/indication_diabetes/dataset/test/output_test.csv')
x_test_padrao = 

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


output_model_ = model.predict(x_test_padrao)
y_test_class = saida_test.values

for i in range(len(output_model_)):
    if(output_model_[i]>= 0.5):
      output_model_[i] = 1
    else:
      output_model_[i] = 0
print(output_model_)


print('Acurácia:', accuracy_score(y_test_class, output_model_))
print('Precisão:', precision_score(y_test_class, output_model_))
print('Sensibilidade:', recall_score(y_test_class, output_model_))
print('F1-Score:', f1_score(y_test_class, output_model_))


cm = confusion_matrix(y_test_class, output_model_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
disp.ax_.set_title('Matriz de confusão')
disp.ax_.set_xlabel('Classificação prevista')
disp.ax_.set_ylabel('Classificação real')
plt.show()