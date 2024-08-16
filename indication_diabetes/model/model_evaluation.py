from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# load model 
model = tf.keras.models.load_model('../Indicador_diabetes/indication_diabetes/model/model.keras')

# load data
saida_test = pd.read_csv('../Indicador_diabetes/indication_diabetes/dataset/test/output_test.csv')
input_test = pd.read_csv('../Indicador_diabetes/indication_diabetes/dataset/test/input_test_standard.csv')

#Carregando o custo
history = pd.read_csv('../Indicador_diabetes/indication_diabetes/model/custo.csv')

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

y_test_class = y_test_class.flatten()
output_model_ = output_model_.flatten()

print(output_model_)
print("y_test:", y_test_class)

for i in range(len(output_model_)):
    if(output_model_[i]>= 0.5):
      output_model_[i] = 1
    else:
      output_model_[i] = 0
print(output_model_)

print('Acurácia:', accuracy_score(y_test_class, output_model_))
print('Precisão:', precision_score(y_test_class, output_model_, average='weighted'))
print('Sensibilidade:', recall_score(y_test_class, output_model_, average='weighted'))
print('F1-Score:', f1_score(y_test_class, output_model_, average='weighted'))

cm = confusion_matrix(y_test_class, output_model_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
disp.ax_.set_title('Matriz de confusão')
disp.ax_.set_xlabel('Classificação prevista')
disp.ax_.set_ylabel('Classificação real')
plt.show()