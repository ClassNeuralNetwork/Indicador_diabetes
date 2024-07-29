import pandas as pd
from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE

db = pd.read_csv("/home/brunopaiva/Diabetes/diabetes_012_health_indicators_BRFSS2021.csv")

# lista=['BMI', 'CholCheck', 'HighChol', 'HighBP', 'Diabetes_012']
# db.columns = lista

db['Diabetes_012'] = db['Diabetes_012'].astype('category').cat.codes

db.head()
db.info()
classes = db['Diabetes_012'].value_counts()
print(classes)

input_train, input_test, output_train, output_test = train_test_split(db.iloc[:, 0:4].values, db['Diabetes_012'].values, test_size=0.2)

input_train = pd.DataFrame(input_train)
input_train.to_csv('input_train_csv', index=False)

input_test = pd.DataFrame(input_test)
input_test.to_csv('input_test_csv', index=False)

output_train = pd.DataFrame(output_train)
output_train.to_csv('output_train_csv', index=False)

output_test = pd.DataFrame(output_test)
output_train.to_csv('output_test_csv', index=False)

# input_train = pd.read_csv('input_train_csv')
# output_train = pd.read_csv('output_train_csv')

# input_train.head()

# sm = SMOTE(random_state=42)

# input_train_balanced, output_train_balanced = sm.fit_resample(input_train, output_train)


# input_train_balanced.to_csv('input_train_balanced_csv', index=False)
# output_train_balanced.to_csv('input_train_balanced_csv', index=False)
