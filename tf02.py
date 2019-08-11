import tensorflow as tf
import pandas as pd

data = pd.read_csv('./dataset/income1.csv')
print(data)

import matplotlib.pyplot as plt
plt.scatter(data.Education, data.Income)