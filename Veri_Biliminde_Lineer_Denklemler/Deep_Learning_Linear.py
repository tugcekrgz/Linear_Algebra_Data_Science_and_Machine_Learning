# Gerekli kütüphanelerin yüklenmesi
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras. layers import Dense
from tensorflow.keras.optimizers import SGD

# Veri setinin yüklenmesi
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Veri ön işleme
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Giriş verisini düzleştirme
x_train = x_train. reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Model oluşturma
model = Sequential( [
Dense(128, activation='relu', input_shape=(784,)), # Gizli katman, ReLU aktivasyonu
Dense(10, activation='softmax') # Çıkış katmanı, 10 sınıf için softmax aktivasyonu
])

# Gradient descent optimizer'in1 kullanarak modeli derleme
sgd = SGD(learning_rate=0.01) # Gradient descent optimizer, öğrenme hızı belirleniyor
model.compile(optimizer=sgd, # Gradient descent optimizer kullanılıyor
        loss='sparse_categorical_crossentropy', # Kayıp fonksiyonu
        metrics=['accuracy']) # Modelin değerlendirilmesi için metrikler

# Modelin eğitimi
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test) )
# Modelin eğitilmesi: 5 epoch boyunca, her seferinde 32 örneklik mini-batch'ler kullanarak

# Modelin değerlendirilmesi
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")