import pandas as pd
from sklearn.model_selection import train_test_split

db = pd.read_csv("/home/brunopaiva/DataSet/indication_diabetes/utils/diabetes_012_health_indicators_BRFSS2021.csv")

y = db['Diabetes_binary'] # Classes/Labels
df = db.drop(['Diabetes_binary'], axis = 1)
x = db.iloc[:, 0:22].values # Atributos/Features

classes = db['Diabetes_binary'].value_counts()
print(classes)

input_train, input_test, output_train, output_test = train_test_split(x, y, test_size=0.2)

input_train = pd.DataFrame(input_train)
input_train.to_csv('input_train_csv', index=False)

input_test = pd.DataFrame(input_test)
input_test.to_csv('input_test_csv', index=False)

output_train = pd.DataFrame(output_train)
output_train.to_csv('output_train_csv', index=False)

output_test = pd.DataFrame(output_test)
output_train.to_csv('output_test_csv', index=False)
