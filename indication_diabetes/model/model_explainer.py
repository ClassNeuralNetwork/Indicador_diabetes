import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.callbacks import EarlyStopping
import shap

# load model 
model = tf.keras.models.load_model('/home/brunopaiva/DataSet/Indicador_diabetes/indication_diabetes/model/model.keras')

# load data 
x_test_padrao = pd.read_csv('/home/brunopaiva/DataSet/Indicador_diabetes/indication_diabetes/dataset/test/input_test_standard.csv') 

lista = ['negativo_diabets','positivo_diabets']

classes = ['Diabetes_binary']

explainer  = shap.KernelExplainer(model.predict, x_test_padrao)
shap_values = explainer.shap_values(x_test_padrao)

shap.summary_plot(shap_values,x_test_padrao,feature_names=lista)
shap.summary_plot(shap_values,x_test_padrao, plot_type="bar" ,feature_names=lista, class_names=classes)

shap.initjs()
shap.force_plot(explainer.expected_value, shap_values, x_test_padrao)