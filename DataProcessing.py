import pandas as pd
import numpy as np

df = pd.DataFrame([ ['green','M','10.1', 'class1'], ['red', 'L', '13.5', 'class2'], ['blue', 'XL', '15.3', 'class1']])
df.columns=['color', 'size','price','class-label']
print(df)

class_mapping={label:idx for idx, label in enumerate(np.unique(df['class-label']))}
print(class_mapping)

df['class-label']=df['class-label'].map(class_mapping)
print(df)

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-database/wine/wine.data')
df_wine.columns = ['Class Label', 'Alcohol', 'Malic Acid', 'Ash', 'Alcalinity of Ash', 'Magnesium', 'Total Phenols',
                   'Flavanoids', 'Non-Flavanoid Phenols', 'Proanthocyannins', 'Color Intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']
print('Class labels', np.unique(df_wine['Class Label']))
df_wine.head()