# https://github.com/atilsamancioglu/MachineLearningNotebooks/blob/main/3-PolynomialRegression.ipynb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

df = pd.read_csv("3-customersatisfaction.csv")

df.drop("Unnamed: 0", axis=1, inplace=True)
print(df.info())

plt.scatter(df["Customer Satisfaction"], df["Incentive"])
plt.xlabel("Customer Satisfaction")
plt.ylabel("Incentive")
plt.show()

# dependent & independent features
X= df[["Customer Satisfaction"]]
y= df["Incentive"]

# train - test split
X_train, X_test, y_train, y_test = train_test_split(X,y)


# scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

regression = LinearRegression()
regression.fit(X_train, y_train)

# prediction
y_pred = regression.predict(X_test)
print(y_pred)

score = r2_score(y_test, y_pred)
print(score)

plt.scatter(X_train, y_train)
plt.plot(X_train, regression.predict(X_train), "r")
plt.show()

poly = PolynomialFeatures(degree=2, include_bias=True)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

print(X_train_poly) # intercept kolonu olan 1'ler eklendi

regression = LinearRegression()
regression.fit(X_train_poly, y_train)
y_pred = regression.predict(X_test_poly)
score = r2_score(y_test, y_pred)
print("poly R2 score: ", score) # R2 skoru kullanılacak degree için fikir verebilir
print("Poly coef_: ", regression.coef_, "   Intercept: ", regression.intercept_)

plt.scatter(X_train, y_train)
plt.scatter(X_train, regression.predict(X_train_poly), color="r")
plt.show()

'''
# with degree=3
poly = PolynomialFeatures(degree=3, include_bias=True)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

regression = LinearRegression()
regression.fit(X_train_poly, y_train)
y_pred = regression.predict(X_test_poly)
score = r2_score(y_test, y_pred)
print("poly degree3 R2 score: ", score)

plt.scatter(X_train, y_train)
plt.scatter(X_train, regression.predict(X_train_poly))
plt.show()
'''

# Yeni Datayla Hesaplama Yapma
new_df = pd.read_csv("3-newdatas.csv")
new_df.rename(columns= {"0": "Customer Satisfaction"}, inplace=True)

X_new = new_df[["Customer Satisfaction"]]
X_new = scaler.fit_transform(X_new)
X_new_poly = poly.fit_transform(X_new)

y_new = regression.predict(X_new_poly)

plt.plot(X_new, y_new, "r", label= "New Predictions")
plt.scatter(X_train, y_train, color="b", label= "Training Points")
plt.scatter(X_test, y_test, color="orange",label= "Test Points")
plt.legend() #shows labels
plt.show()

# Pipeline
def poly_regression(degree):
    poly_features = PolynomialFeatures(degree=degree)
    lin_reg = LinearRegression()
    scaler = StandardScaler()
    pipeline = Pipeline(
        [("standart_scaler", scaler),
         ("poly_feature", poly_features),
         ("lin_reg", lin_reg)]
    )
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print(f"Pipeline R2 score of degree({degree}): {score}")
    
    y_pred_new = pipeline.predict(X_new)
    plt.plot(X_new, y_pred_new, "r", label="New Predictions")
    plt.scatter(X_train, y_train, label="Training Points")
    plt.scatter(X_test, y_test, label="Test Points")
    plt.title(f"Graph of degree( {degree} ) with R2 score of {score}")
    plt.legend()
    plt.show()


# verilen derecelerin r2 sonuçları kodun üst satırlarında hesapladığımız r2 skorları ile aynı
poly_regression(1)
poly_regression(2)
poly_regression(3)

# degree arttıkça predictionların overfitting'i artıyor
for degre in range(3,11):
    poly_regression(degre)
