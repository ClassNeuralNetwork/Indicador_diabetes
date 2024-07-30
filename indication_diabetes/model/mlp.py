import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import tensonflow as tf

input_train = pd.read_csv('/home/brunopaiva/DataSet/indication_diabetes/dataset/train/input_train_csv')
output_train = pd.read_csv('/home/brunopaiva/DataSet/indication_diabetes/dataset/train/output_train_csv')

model = tf.keras.models.Squential()
model.add(tf.keras.layers.Dense(units=28))