import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping

x_treino_padrao = pd.read_csv('/home/brunopaiva/DataSet/Indicador_diabetes/indication_diabetes/dataset/train/input_train_balanced.csv')
saida_treino = pd.read_csv('/home/brunopaiva/DataSet/Indicador_diabetes/indication_diabetes/dataset/train/output_train_balanced.csv')
x_treino_padrao.info()

# e camada de saída com 2 neurônios (2 classes)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(500, input_dim=21, kernel_regularizer=tf.keras.regularizers.L2(0.01), activation='sigmoid', name='oculta'))
# model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(256, activation='relu', input_shape=(21,)))

# Camada oculta 2 com 400 neurônios
model.add(tf.keras.layers.Dense(128, activation='relu'))

# Camada oculta 3 com 300 neurônios
model.add(tf.keras.layers.Dense(64, activation='relu'))

model.add(tf.keras.layers.Dense(32, activation='relu'))

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
history = model.fit(x_treino_padrao, saida_treino, validation_split=0.2, epochs=50, callbacks=[early_stopping])


model.save('model.keras')