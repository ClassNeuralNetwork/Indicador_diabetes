import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping

#Carregar os dados de treinamento
input_train = pd.read_csv('../Indicador_diabetes/indication_diabetes/dataset/train/input_train_balanced.csv')
output_train = pd.read_csv('../Indicador_diabetes/indication_diabetes/dataset/train/output_train_balanced.csv')

#Carregar os dados de teste
input_test = pd.read_csv('../Indicador_diabetes/indication_diabetes/dataset/test/input_test.csv')
output_test = pd.read_csv('../Indicador_diabetes/indication_diabetes/dataset/test/output_test.csv')

#Normalizando os dados
scaler = StandardScaler()
input_train_scaled = scaler.fit_transform(input_train)
input_test_scaled = scaler.transform(input_test)

#Salvando os dados normalizados
pd.DataFrame(input_train_scaled).to_csv('../Indicador_diabetes/indication_diabetes/dataset/train/input_train_standard.csv', index=False)
pd.DataFrame(input_test_scaled).to_csv('../Indicador_diabetes/indication_diabetes/dataset/test/input_test_standard.csv', index=False)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(256, activation='relu', input_shape=(input_train_scaled.shape[1],)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.1))
# model.add(tf.keras.layers.Dense(64, activation='relu'))

model.add(tf.keras.layers.Dense(1, activation='sigmoid', name='saida'))
model.summary() #visualizando o modelo

# Compile model
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train model
history = model.fit(input_train_scaled, output_train, validation_split=0.2, batch_size=32, epochs=100, callbacks=[early_stopping])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

pd.DataFrame(history.history).to_csv('../Indicador_diabetes/indication_diabetes/model/custo.csv', index=False)

model.save('../Indicador_diabetes/indication_diabetes/model/model.keras')