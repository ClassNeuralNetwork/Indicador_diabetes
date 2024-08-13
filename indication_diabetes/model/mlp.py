import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping

# x_treino_padrao = pd.read_csv('/home/brunopaiva/DataSet/Indicador_diabetes/indication_diabetes/dataset/train/input_train_balanced.csv')
# saida_treino = pd.read_csv('/home/brunopaiva/DataSet/Indicador_diabetes/indication_diabetes/dataset/train/output_train_balanced.csv')
# x_treino_padrao.info()

#Carregar os dados
input_train = pd.read_csv('/home/brunopaiva/DataSet/Indicador_diabetes/indication_diabetes/dataset/train/input_train.csv')
output_train = pd.read_csv('/home/brunopaiva/DataSet/Indicador_diabetes/indication_diabetes/dataset/train/output_train.csv')
input_test = pd.read_csv('/home/brunopaiva/DataSet/Indicador_diabetes/indication_diabetes/dataset/test/input_test.csv')
output_test = pd.read_csv('/home/brunopaiva/DataSet/Indicador_diabetes/indication_diabetes/dataset/test/output_test.csv')

#Normalizando os dados
scaler = StandardScaler()
input_train_scaled = scaler.fit_transform(input_train)
input_test_scaled = scaler.transform(input_test)

#DataFrame com os nomes das colunas
#input_train_standard = pd.DataFrame(input_train_scaled)

#Salvando os dados normalizados
pd.DataFrame(input_train_scaled).to_csv('/home/brunopaiva/DataSet/Indicador_diabetes/indication_diabetes/dataset/train/input_train_standard.csv', index=False)
pd.DataFrame(input_test_scaled).to_csv('/home/brunopaiva/DataSet/Indicador_diabetes/indication_diabetes/dataset/test/input_test_standard.csv', index=False)


# e camada de saída com 2 neurônios (2 classes)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(500, input_dim=21, kernel_regularizer=tf.keras.regularizers.L2(0.01), activation='sigmoid', name='oculta'))
# model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(256, activation='relu', input_shape=(input_train_scaled.shape[1],)))
model.add(tf.keras.layers.Dropout(0.1))
# Camada oculta 2 com 400 neurônios
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.1))
# Camada oculta 3 com 300 neurônios
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.1))

model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dropout(0.1))

# Camada de saída (exemplo para classificação binária)
# model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.add(tf.keras.layers.Dense(1, activation='sigmoid', name='saida'))
model.summary() #visualizando o modelo

# Compile model
# Otimizador Adam com taxa de aprendizado de 0.01
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
# Função de custo categorical_crossentropy (para problemas de classificação com mais de duas classes)
# Métrica de avaliação MSE (Mean Squared Error)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train model
history = model.fit(input_train_scaled, output_train, validation_split=0.2, epochs=50, callbacks=[early_stopping])

model.save('/home/brunopaiva/DataSet/Indicador_diabetes/indication_diabetes/model/model.keras')

pd.DataFrame(history.history).to_csv('/home/brunopaiva/DataSet/Indicador_diabetes/indication_diabetes/model/custo.csv')