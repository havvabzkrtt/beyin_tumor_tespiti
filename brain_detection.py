import warnings 
warnings.filterwarnings("ignore")  # hatalari yok sayar
import tensorflow as tf  # derin sinir aglarinda egitim ve ogrenim
import pandas as pd  #veri islemi ve analizi
import matplotlib.pyplot as plt  # veri gorsellestirme, grafik
import cv2 as cv # goruntu isleme
import numpy as np # matris islemleri
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from sklearn.model_selection import train_test_split # test ve egitim verilerini ayirma
import keras.layers as tfl
from keras.models import Sequential
from keras.layers import Flatten,Activation,Dense,Dropout,Conv2D,MaxPool2D
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

# Egitimde kullanlacak verilerin classslar.

# {'glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor'}

# egitim ve test verilerinim cekilmesi
tumorTestPath = "brain_tumor/Testing/"
tumorTrainPath = "brain_tumor/Training/"

# ImageDataGenerator => veriden en iyi sekilde yararlanmak icin, rastgele donusumle veriyi arttiriyoruz
# rescale => islemi kolaylastirmak icin 0-255 arasindaki degerleri 0-1 arasindaki degerler haline getirir.
trainGen = ImageDataGenerator(rescale=1./255)

# verilerin islenmesi icin uygun hale getirilmesi
tumorTest = trainGen.flow_from_directory(
        tumorTestPath,
        target_size=(200 , 200),
        batch_size=32)

tumorTrain= trainGen.flow_from_directory(
        tumorTrainPath,  target_size=(200 , 200),
        batch_size=32)

# target_size => goruntuyu yeninden boyutlandirir (yukseklik,genislik)
# batch_size => bir ornek gruptaki veri saysini

#model
model =Sequential()
# sirali bir model, katmanlar modele tek tek eklenir
#convolution and maxpoollayer

model.add(Conv2D(filters=10,kernel_size=3,
                input_shape=(200,200,3)))

# Conv2D => konvonusyonel katman
# filters => konvonusyonel katmanninda ogrenecegi filtre sayisi
# input_shape => girislerin boyutunu tutar

model.add(Activation('relu'))

model.add(MaxPool2D(pool_size=2))
# pool_size => 2,2 matris alarak iþleme sokar
#flatten layer

model.add(Flatten())
# düzenleþtirme katmani en onemli katman olan Fully Connected Layer'in giriþindeki verileri hazirlar.

#hidden layer
model.add(Dense(16))
# bir onceki katmandaki tüm dügümler mecut katmandaki dügümlere baglanir
# units => 16 / dügüm sayisi

model.add(Activation('relu'))

#output layer
model.add(Dense(4))
model.add(Activation('sigmoid'))

optimizer = keras.optimizers.legacy.Adam(lr = 0.0001) #0.001
# optimizer => w(agirlik) degerlerinin optimize edilmesi icin kullanilir.
# lr => overfittingi engellemek icin kullanilan katsayi 

model.compile(optimizer= optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
# compile => modelin kullanima hazir hale getirilmesi icin derlenmesi gerekir. 
# Derleme icin bir optimize edici ve bir kayip fonksiyonu belirlememiz gerekir
# loss => egitim sonunda elde edilen deger ile gercek deger arasindaki hata farkinin hesaplanmasi
# metrics => modelin baþarisini incelemek icin kullanilan metrik

#training the model
early_stop = EarlyStopping(monitor="val_loss", patience = 3)
# izlenen metrik iyileþmeyi durdurunca egitimi durdurulmali
# monitor => izlenemesi gerek miktar
# patience => egitimin durdurulacagi iyileþtirme olmayan epoch sayisi

history = model.fit_generator(tumorTrain, callbacks=[early_stop], validation_data=tumorTest,epochs=15)
# fit_generator =>> gercek zamanli veri aktarimini saglar, bellege sigdirilmasi gereken cok büyük veri kümesi oldugunda 
# tumorTrain => alinacak egitim verisi
# validation_data => dogrulama verileri, test verileri verildi
# epoch => modelin egitilip duruma gore agirliklarin güncellendigi her adim

# grafik cizdirme
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()