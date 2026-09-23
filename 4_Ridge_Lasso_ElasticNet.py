# https://github.com/atilsamancioglu/MachineLearningNotebooks/blob/main/4-RidgeLassoElasticNet.ipynb 
#https://www.kaggle.com/datasets/nitinchoudhary012/algerian-forest-fires-dataset/data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV #CROSS VALIDATION


df = pd.read_csv('4-Algerian_forest_fires_dataset.csv')

print(df.info())
print(df.isnull().sum())
print(df[df.isnull().any(axis=1)]) # null satırları göster

df.drop(122, inplace=True)
# kolon 123te bölge adı ile bölgeler ayrılmış. 0 ve 1 ile bölgelerin ayrımını belirtelim
df.loc[:123, "Region"] = 0
df.loc[123:, "Region"] = 1


df = df.dropna().reset_index(drop=True)
# print(df.iloc[121], df.iloc[122])

# print( df.columns )
df.columns = df.columns.str.strip() # bazı kolon adlarında ki boşlukları sil
# print( df.columns )

print( df["day"] == "day") # bu kolon silinecek
df.drop(122, inplace=True)

# object türündeki kolonlar int ve float olarak değiştir
df[["day", "month", "year", "Temperature", "RH", "Ws"]] = df[["day", "month", "year", "Temperature", "RH", "Ws"]].astype(int)
df[["Rain", "FFMC", "DMC", "ISI", "BUI", "FWI"]] = df[["Rain", "FFMC", "DMC", "ISI", "BUI", "FWI"]] .astype(float)

# fire ve not fire değerlerini uygun şekilde değiştir
print(df["Classes"].unique(), df["Classes"].value_counts()) 
df["Classes"] = np.where( df["Classes"].str.contains("not fire"), 0,1 ) # not fire içerenleri 0 ile içermeyenleri 1 ile değiştirir
print(df["Classes"].unique(), df["Classes"].value_counts())
# .value_counts(normalize=True) ..içeriklerin yüzdesini verir

'''
sns.heatmap(df.corr(), annot=True)
plt.show()
'''

df.drop(["day", "month", "year"], axis=1, inplace=True)

# Dependent & independent features
X = df.drop("FWI", axis=1)
y = df["FWI"]

#train - test split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=15)

print( X_train.corr() )
# redundancy: iki kolonun korelasyonunun çok yüksek olması bunların birbiriyle bağlantılı olduğu ve bilgi tekrarı yaptığını gösterir (Redundant). 
#             Bu kolonlar modeli kompleks hale getirir. bu kolonlar modelden atılabilir.
# multicollinearity (çoklu doğrusal bağlantı) => overfitting

def correlation_for_dropping(df, threshold):
    columns_to_drop = set()
    corr = df.corr()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i,j]) > threshold:
                columns_to_drop.add(corr.columns[i])
    return columns_to_drop

X_train.drop(correlation_for_dropping(X_train, 0.85), axis=1, inplace=True)
X_test.drop(correlation_for_dropping(X_test, 0.85), axis=1, inplace=True)
print(X_train.shape, X_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
'''
plt.subplots(figsize=(15,5))
plt.subplot(1,2,1)
sns.boxplot(data=X_train)
plt.title("X_train")
plt.subplot(1,2,2)
sns.boxplot(data=X_train_scaled)
plt.title("X_train_scaled")
plt.show() # scale halinin değişimini incelemek için
'''

linear = LinearRegression()
linear.fit(X_train_scaled, y_train)
y_pred = linear.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("Mean Absolute Error: ", mae)
print("Mean Squared Error: ", mse)
print("R2 Score: ", score)
plt.scatter(y_test,y_pred)
plt.show()

lasso = Lasso()
lasso.fit(X_train_scaled, y_train)
y_pred = lasso.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("Mean Absolute Error: ", mae)
print("Mean Squared Error: ", mse)
print("R2 Score: ", score)
plt.scatter(y_test,y_pred)
plt.show()

ridge = Ridge()
ridge.fit(X_train_scaled, y_train)
y_pred = ridge.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("Mean Absolute Error: ", mae)
print("Mean Squared Error: ", mse)
print("R2 Score: ", score)
plt.scatter(y_test,y_pred)
plt.show()

elasticnet = ElasticNet()
elasticnet.fit(X_train_scaled, y_train)
y_pred = elasticnet.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("Mean Absolute Error: ", mae)
print("Mean Squared Error: ", mse)
print("R2 Score: ", score)
plt.scatter(y_test,y_pred)
plt.show()

# LASSO CROSS VALIDATION

lassocv = LassoCV(cv=5)

lassocv.fit(X_train_scaled, y_train)
y_pred = lassocv.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("Mean Absolute Error: ", mae)
print("Mean Squared Error: ", mse)
print("R2 Score: ", score)
plt.scatter(y_test,y_pred)
plt.show()
print(f'Alpha Value for LassoCV [lassocv.alpha_] = {lassocv.alpha_}')
print(f'Alphas Value for LassoCV [lassocv.alpha_] = {lassocv.alphas_}')

# RIDGE CROSS VALIDATION
ridgecv = RidgeCV()

ridgecv.fit(X_train_scaled, y_train)
y_pred = ridgecv.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("Mean Absolute Error: ", mae)
print("Mean Squared Error: ", mse)
print("R2 Score: ", score)
plt.scatter(y_test,y_pred)
plt.show()

# ELASTIC NET CROSS VALIDATION

elasticnetcv = ElasticNetCV()

elasticnetcv.fit(X_train_scaled, y_train)
y_pred = elasticnetcv.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("Mean Absolute Error: ", mae)
print("Mean Squared Error: ", mse)
print("R2 Score: ", score)
plt.scatter(y_test,y_pred)
plt.show()
