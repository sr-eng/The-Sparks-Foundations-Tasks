import csv
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

File = open('datatask1.csv','r')
x = list()
y = list()
lines = csv.reader(File)
for line in lines:
    x.append([float(line[0])])
    y.append(int(line[1]))
model = LinearRegression()
model.fit(x,y)
prediction = model.predict(([[9.25]]))
print("Predicted Score = ",prediction)
line = model.coef_*x+model.intercept_
plt.scatter(x, y)
plt.plot(x, line)
plt.show()

