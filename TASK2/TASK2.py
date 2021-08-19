# task 2
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

iris = datasets.load_iris()
print(iris.data)
x = pd.DataFrame(iris.data)
y = pd.DataFrame(iris.target)
x.columns = ['Sepal_Length', 'Sepal_Width','Petal_Length','Petal_Width']
y.columns = ['Targets']
# K-means
model = KMeans(n_clusters=3)
model.fit(x)
colors = np.array(['Black','Blue','Purple'])
plt.scatter(x.Petal_Length,x.Petal_Width,s=40,c=colors[model.labels_])
plt.show()

