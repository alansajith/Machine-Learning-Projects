import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# importing dataset
dataset = pd.read_csv(
    r"C:\Users\Alansajith\OneDrive\Desktop\Machine learning\Part 2 - Regression\Section 6 - Polynomial Regression\Python\Position_Salaries.csv"
)
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# training on linear regression on the whole dataset
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x, y)

# training on polynomial regression on the whole dataset
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=13)
x_poly = poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

# visual representation of linear model on the dataset
plt.scatter(x, y, color="red")
plt.plot(x, lin_reg.predict(x), color="blue")
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()


# visual representation of polynomial regression  model on the dataset
plt.scatter(x, y, color="red")
plt.plot(x, lin_reg2.predict(x_poly), color="blue")
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()
