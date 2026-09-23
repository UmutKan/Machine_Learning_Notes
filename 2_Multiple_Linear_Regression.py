# https://github.com/atilsamancioglu/MachineLearningNotebooks/blob/main/2-MultipleLinearRegression.ipynb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("D:\\umutUni\\Veri Bilimi & Makine Öğrenmesi YZ Kursu\\Machine Learning\\2-multiplegradesdataset.csv")
print(df.isnull().sum(), df.corr())

# sns.pairplot(df)
# plt.show()

# sns.regplot( x= df["Study Hours"], y= df["Exam Score"] )
# plt.show()

# sns.regplot(x=df['Sleep Hours'],y=df['Exam Score'])
# plt.show()

# sns.regplot(x=df['Social Media Hours'],y=df['Exam Score'])
# plt.show()

# independent and dependent features
'''
X = df.iloc[:,:-1] # son kolon hariç hepsini atar
y = df.iloc[:,-1] # son kolonu atar
'''
X = df[["Study Hours", "Sleep Hours", "Attendance Rate", "Social Media Hours"]]
y = df["Exam Score"]

# train - test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state=15)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

regression = LinearRegression()
regression.fit(X_train, y_train)

new_student = [[5, 7, 90, 2]] # [[Study Hours, Sleep Hours, Attendance Rate, Social Media Hours]]
new_student_scaled = scaler.transform(new_student)

print("Predicted Exam Score of new_student: ", regression.predict(new_student_scaled))

# prediction
y_pred = regression.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("mse: ", mse, "   mae: ", mae)

score = r2_score(y_test, y_pred)
adjusted_r2_score = 1 - (1-score)*( len(y_test)-1 )/ (len(y_test)-X_test.shape[1]-1)
print("R2 score: ", score, "   Adjusted R2:", adjusted_r2_score)

# plt.scatter(y_test, y_pred) # Çıkan grafik doğrusal ise aldığımız sonucun mantıklı olduğu söylenebilir
# plt.show()

# residuals = y_test - y_pred
# sns.displot(residuals, kind="kde")
# plt.show()

print("Intercept: ", regression.intercept_, "   Coefficient: ", regression.coef_)
