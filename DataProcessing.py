import pandas as pd
import numpy as np

df = pd.DataFrame([ ['green','M','10.1', 'class1'], ['red', 'L', '13.5', 'class2'], ['blue', 'XL', '15.3', 'class1']])
df.columns=['color', 'size','price','class-label']
print(df)

class_mapping={label:idx for idx, label in enumerate(np.unique(df['class-label']))}
print(class_mapping)

df['class-label']=df['class-label'].map(class_mapping)
print(df)