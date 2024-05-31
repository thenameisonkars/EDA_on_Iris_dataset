import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
from sklearn.datasets import load_iris
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Data inspection
print(iris_df.info())
print(iris_df.describe())

# Data visualization
# Univariate analysis
iris_df.hist(figsize=(10,8))
plt.show()

# Bivariate analysis
sns.pairplot(iris_df)
plt.show()

# Multivariate analysis
corr = iris_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

from google.colab import drive
drive.mount('/content/drive')
