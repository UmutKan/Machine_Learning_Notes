# https://github.com/atilsamancioglu/MachineLearningNotebooks/blob/main/1-SimpleLinearRegression.ipynb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("D:\\umutUni\\Veri Bilimi & Makine Öğrenmesi YZ Kursu\\Machine Learning\\1-studyhours.csv")

# plt.scatter(df["Study Hours"], df["Exam Score"])
# plt.xlabel("Study Hours")
# plt.ylabel("Exam Score")
# plt.show()

# independent(X) and dependent(y) features
X = df[["Study Hours"]] # X'in data frame olması gerekli bu sebeple iki [] kullanılır veya pd.DataFrame de çalışır
y = df["Exam Score"]

# Test - Train Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 15)

# Standardize data set
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) # Test setine sadece transform uygulanmalıdır

regression = LinearRegression() # n_jobs = -1 kullanrak cpu'nun çoğunluğunu kullandırır uzun süren öğrenmelerde kullanılabilir
regression.fit(X_train, y_train)

print("Coefficient [tetha_1]: ", regression.coef_) # Coefficient [tetha_1]:  [16.17860223] 
print("Intercept [tetha_0]: ", regression.intercept_) # Intercept [tetha_0]:  76.9076923076923
# yani denklemimiz y = 76.91 + 16.18 x

plt.scatter(X_train, y_train)
plt.plot(X_train, regression.predict(X_train), "r")
plt.show()

# x= 20 , y= ?
# [[]] kullanıldı çünkü method data frame istiyor
y_when_x_is_20 = regression.predict( scaler.transform([[20]]) ) # scaler.transform ile x=20 iken z'nin değerini bulduk (standart normal dağılım)
print("20 saat çalışan kişinin muhtemel sınav sonucu: ", y_when_x_is_20)

# Prediction with test data
y_pred_test = regression.predict(X_test)

mse = mean_squared_error(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
print( "mse: ",  mse, "\nmase: ", mae, "\nrmse: ", rmse)

r2 = r2_score(y_test, y_pred_test)
print("r2 score: ", r2)

adjusted_r2_score = 1 - (1-r2)*( len(y_test)-1 )/ (len(y_test)-X_test.shape[1]-1)
print("Adjusted r2 score: ", adjusted_r2_score)