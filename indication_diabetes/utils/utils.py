import pandas as pd
from sklearn.model_selection import train_test_split

db = pd.read_csv("/home/brunopaiva/DataSet/Indicador_diabetes/indication_diabetes/dataset/diabetes_binary_health_indicators_BRFSS2021.csv")

y = db['Diabetes_binary'] # Classes/Labels
df = db.drop(['Diabetes_binary'], axis = 1)
x = db.iloc[:, 0:22].values # Atributos/Features

classes = db['Diabetes_binary'].value_counts()
print(classes)

input_train, input_test, output_train, output_test = train_test_split(x, y, test_size=0.2)

input_train = pd.DataFrame(input_train)
input_train.to_csv('/home/brunopaiva/DataSet/Indicador_diabetes/indication_diabetes/dataset/train/input_train.csv', index=False)

input_test = pd.DataFrame(input_test)
input_test.to_csv('/home/brunopaiva/DataSet/Indicador_diabetes/indication_diabetes/dataset/test/input_test.csv', index=False)

output_train = pd.DataFrame(output_train)
output_train.to_csv('/home/brunopaiva/DataSet/Indicador_diabetes/indication_diabetes/dataset/train/output_train.csv', index=False)

output_test = pd.DataFrame(output_test)
output_train.to_csv('/home/brunopaiva/DataSet/Indicador_diabetes/indication_diabetes/dataset/test/output_test.csv', index=False)
