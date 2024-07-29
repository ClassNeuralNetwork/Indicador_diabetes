import pandas as pd
from imblearn.over_sampling import SMOTE
input_train = pd.read_csv('input_train_csv')
output_train = pd.read_csv('output_train_csv')

sm = SMOTE(random_state=42)
input_train_balanced, output_train_balanced = sm.fit_resample(input_train, output_train)

print(output_train_balanced.value_counts())
# input_train.head()
input_train_balanced.to_csv('input_train_balanced_csv', index=False)
output_train_balanced.to_csv('output_train_balanced_csv', index=False)