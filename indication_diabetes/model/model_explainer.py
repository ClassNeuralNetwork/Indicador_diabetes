import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import shap

# Carregar o modelo
model = tf.keras.models.load_model('../Indicador_diabetes/indication_diabetes/model/model.keras')

selected_feature = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 
                    'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 
                    'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'age', 'Education', 'Incone']

# Carregar dados normalizados
input_test = pd.read_csv('../Indicador_diabetes/indication_diabetes/dataset/test/input_test_standard.csv', header=0, names=selected_feature)
input_train = pd.read_csv('../Indicador_diabetes/indication_diabetes/dataset/train/input_train_standard.csv', header=0, names=selected_feature)

# Usando shap.sample para reduzir o tamanho dos dados de fundo
background_data = shap.sample(input_test, 100)

# Criando o KernelExplainer com background_data reduzido
explainer = shap.KernelExplainer(model.predict, background_data, nsamples=100)

# Calcular os valores de shap para uma amostra de 10 pontos de dados de teste
shap_values = explainer.shap_values(input_test.iloc[:100])

# Plotar o resumo dos valores de shap
plt.figure()
shap.summary_plot(shap_values, input_test.iloc[:100], plot_type="bar", feature_names=input_test.columns)

# Plotar o gráfico de dependência
plt.figure()
shap.dependence_plot(selected_feature[0], shap_values, input_test.iloc[:100], feature_names=input_test.columns)
