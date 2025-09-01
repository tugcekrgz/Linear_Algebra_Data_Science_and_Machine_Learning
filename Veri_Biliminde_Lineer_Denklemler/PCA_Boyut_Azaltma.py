import numpy as np 

#Örnek veri oluşturma
np.random.seed(42)

data={
    'X1': np. random. rand(100),
    'X2': np. random. rand(100),
    'X3': np. random. rand(100),
    'X4': np. random. rand(100),
    'X5': np. random. rand(100),
    'Y': np. random. rand(100)
    
}

import pandas as pd

df= pd.DataFrame(data)

# 1. Veri Setinin Standartlaştırılması

for column in df.columns[ :- 1]: # 'Y' hariç tum sutunlar uzerinde don
    mean_value = np.mean(df [column] )
    std_dev = np.std(df[column])
    df[column] = (df[column] - mean_value) / std_dev

# Adım 2: Kovaryans Matrisinin Hesaplanması

# Veri setindeki standartlaştırılmış bağımsız değişkenleri seçme
X = df.drop('Y', axis=1)

# Standartlaştırılmış veri üzerinden kovaryans matrisini hesapla
kovaryans_matrisi = (X.T @ X) / len(X)

# Adım 3: Kovaryans Matrisinin Özdeğer ve Özvektörlerinin Bulunması
ozdegerler, ozvektorler = np.linalg.eig(kovaryans_matrisi)

# Adım 4: Özdeğerlerin Sıralanması ve Özvektörlerin Seçimi

# Özdeğerleri ve özvektörleri büyükten küçüğe sıralama
sirali_indexler = np.argsort(ozdegerler) [ ::- 1]
sirali_ozdegerler = ozdegerler[sirali_indexler]
sirali_ozvektorler = ozvektorler[sirali_indexler]

# İstenen boyutta özdegerlere karşılık gelen özvektörleri seçme

yeni_boyut = 2 # İstenen yeni boyut
principal_components = sirali_ozvektorler[: yeni_boyut]

# Adım 5: Yeni Değişken Matrisinin Oluşturulması

# Seçilen princpical componentlar ile yeni değişken matrisini oluşturma
yeni_veri = X @ principal_components.T

# Adım 6: Boyut İndirgenmiş Veri Setini Oluşturma
boyut_indirgenmis_veri = pd.concat([yeni_veri, df['Y']], axis=1, ignore_index=True)

boyut_indirgenmis_veri.head()

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

np. random. seed (42)

data = {
    'X1': np.random. rand(100),
    'X2': np.random. rand(100),
    'X3': np.random. rand(100),
    'X4': np. random. rand (100),
    'X5': np.random. rand(100),
    'Y': np.random. rand (100)
}

X = df.drop('Y', axis=1)

# PCA modelini oluşturma
pca = PCA(n_components=2) # Boyut azaltmayı 2 bileşene yapacak şekilde ayarlandı

# Veriyi PCA'ya uygulama ve azaltılmış boyutlu veri elde etme
new_data = pca.fit_transform(X)

dimensional_reduced_df = pd.concat([pd.DataFrame(new_data), df['Y']], axis=1, ignore_index=True)

dimensional_reduced_df.head()