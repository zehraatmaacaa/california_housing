# California Housing Veri Setini Kullanarak İki Farklı Algoritmayla Ev Fiyatları Tahmini
# Lineer Regresyon ve Karar Ağacı kullandım

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

california = fetch_california_housing() 

#Ev fiyatları ve özelliklerini ayırdık
X = pd.DataFrame(california.data, columns=california.feature_names)
y = pd.Series(california.target)

#Verilerimizi kontrol ediyoruz
print(X.info())

print(X.isnull().sum())  #Eksik veri var mı? Medyan ve ortalamayla doldurabilirdik fakat eksik veri yok.

#Eğitim ve test verilerine ayırıyoruz.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Lineer Regresyon Modeli
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

#Karar Ağacı Regresyonu Modeli
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

#Tahmin
y_pred_linear = linear_model.predict(X_test)
y_pred_tree = tree_model.predict(X_test)

#Performans değerlendirme(R2 skoru ve MSE)
print("Lineer Regresyon Modeli:")
print("R2 Skoru: ", r2_score(y_test, y_pred_linear))
print("MSE: ", mean_squared_error(y_test, y_pred_linear))

print("\nKarar Ağacı Modeli:")
print("R2 Skoru: ", r2_score(y_test, y_pred_tree))
print("MSE: ", mean_squared_error(y_test, y_pred_tree))
#Sonuçları karşılaştırmak için görselleştiriyourz.
plt.figure(figsize=(10, 6))

#Lineer Regresyondan elde ettiğimiz sonuçlar
plt.scatter(y_test, y_pred_linear, color='blue', label='Lineer Regresyon', alpha=0.6)

#Karar Ağacı algoritmasından elde ettiğimiz sonuçlar
plt.scatter(y_test, y_pred_tree, color='red', label='Karar Ağacı', alpha=0.6)

#Gerçek ev fiyatları ile tahmin edilen fiyatları karşılaştıran doğru
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', lw=2)

plt.xlabel("Gerçek Ev Fiyatları")
plt.ylabel("Tahmin Edilen Ev Fiyatları")
plt.title("Ev Fiyatları Tahmini: Lineer Regresyon vs Karar Ağacı")
plt.legend()
plt.show()
