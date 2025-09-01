import numpy as np 

#Rastgele sayı üretiminde kullanılacak seed değerinin ayarlanması (tekrarlanabilirlik içim)
np.random.seed(42)

#100 elemanlı rastgele X1,X2,X3 ve Y değerleri içeren bir veri kümesi oluşturma

data= {
    'X1':np.random.rand(100),
    'X2':np.random.rand(100),
    'X3':np.random.rand(100),
    'Y':np.random.rand(100)
}

#X1,X2,X3 ' ü birleştirerek X matrisini oluşturma

X=np.column_stack((data['X1'],data['X2'],data['X3']))

#Hedef değişken (Y) matrisini oluşturma
Y=data['Y']

# Sabit terimi (beta_0) hesaplamak için X matrisine bir sütun ekleme
X = np.column_stack((np.ones(len(X)), X))

#1.yol: En küçük kareler yöntemi ile katsayıları hesapla
beta = np.matmul(np.matmul(np. linalg.inv(np.matmul(X_T, X)), X_T), Y)

#2.yol: En küçük kareler yöntemi ile katsayıları hesapla
beta = np. linalg. inv(X.T @ X) @ X.T @ Y

# Elde edilen regresyon katsayılarını ekrana yazdırma
print("Regresyon Katsayıları:")
print("beta_0 (sabit terim):", beta[0])
print("beta_1:", beta[1])
print("beta_2:", beta[2])
print("beta_3:", beta[3])

# Fonksiyon: En Küçük Kareler Yöntemi
def En_Kucuk_Kareler_Yontemi(X, Y):
# X matrisine bir sütun ekleme
    X = np.column_stack((np.ones(len(X)), X))

    # Katsayıları hesapla
    beta = np. linalg. inv(X.T @ X) @ X.T @ Y

    return beta

np. random.seed(42)

data = {
'X1': np.random. rand(100),
'X2': np.random. rand(100),
'X3': np.random. rand(100),
'Y': np.random. rand(100)
}
X = np.column_stack((data['X1'], data['X2'],data['X3']))
Y = data ['Y' ]   

katsayilar = En_Kucuk_Kareler_Yontemi(X, Y)

# Elde edilen regresyon katsayılarını ekrana yazdırma
print("Regresyon Katsayıları:")
print("beta_0 (sabit terim):", katsayilar[0])
print("beta_1:", katsayilar[1])
print("beta_2:", katsayilar[2])
print("beta_3:", katsayilar [3])

import pandas as pd
from sklearn. linear_model import LinearRegression

np. random. seed (42)

data = {
'X1': np.random. rand(100),
'X2': np.random. rand(100),
'X3': np.random. rand(100),
'Y': np. random. rand (100)
}

# Pandas DataFrame'i oluşturma
df = pd. DataFrame(data)

# Bağımsız değişkenleri (X) ve bağımlı değişkeni (Y) ayırma
X = df [['X1', 'X2', 'X3' ]]
Y = df ['Y']

# Scikit-Learn kütüphanesi ile lineer regresyon modeli oluşturma
model = LinearRegression()

# Modeli eğitme
model.fit(X, Y)

# Elde edilen kesim noktasını (intercept) yazdır
print(model.intercept_)

# Elde edilen katsayıları yazdır
print(model.coef_)