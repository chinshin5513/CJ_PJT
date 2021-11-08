import numpy as np
import matplotlib.pyplot as plt

# 데이터 로드하기
X = np.load('./211024_rgb_cnn_input.npy')
Y = np.load('./211024_rgb_cnn_target.npy')
print("인풋 세트의 배열형태: ", X.shape)
print("타겟 세트의 배열형태: ", Y.shape)

# 데이터의 구성을 알아보는 것은 이전과 동일한 작업
x = np.unique(Y, return_counts=True)
damage_class = int(x[0][0])
normal_class = int(x[0][1])
damage_count = int(x[1][0])
normal_count = int(x[1][1])

print("손상 컨테이너: {}장".format(damage_count))
print("정상 컨테이너: {}장".format(normal_count))

total = damage_count + normal_count
d_ratio = damage_count/total*100
n_ratio = normal_count/total*100
print("손상 사진 비율: {0:.2f}%, 정상 사진 비율: {1:.2f}%".format(d_ratio, n_ratio))

# 주어진 데이터를 훈련 세트와 테스트 세트로 나누어보자.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, shuffle =True, random_state=42)

# 255로 나눠주어 전처리 시작(그림 데이터, RGB 3차원 데이터)
my_size = 128
X_train = X_train.reshape(X_train.shape[0], my_size, my_size, 3).astype('float32')/255
X_test = X_test.reshape(X_test.shape[0], my_size, my_size, 3).astype('float32')/255
print("훈련 세트 배열: ", X_train.shape)
print("테스트 세트 배열: ", X_test.shape)

# 딥러닝에 필요한 라이브러리 불러오기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import tensorflow as tf
np.random.seed(3)
tf.random.set_seed(3)

# 컨볼루션 층을 어마무시하게 쌓고, 필터 사이즈를 줄이니깐 상당히 높은 정확도가 나타났다.
# CNN은 데이터가 컨볼루션 층을 거치면서 데이터의 특징을 뽑아내며 줄여가는 것이 특징인데...결과는 이렇게 나타났네?
# 해당 모델의 정확도는 92.58% 수준, 손실은 4.658 수준
# 층이 하나일 때는 정확도 89.25% 수준, 손실은 0.4675 수준
model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 2), input_shape=(128, 128, 3), activation='relu'))
model.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
model.add(Conv2D(256, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid')) 
model.summary()

# 모델 컴파일하기
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Best 모델 저장하기, 개선 없으면 훈련 중단하기 옵션 설정하기
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
from keras.callbacks import ModelCheckpoint
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

# 모델 훈련하기
history2 = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=50, batch_size=64, callbacks=[es, mc])  

# 테스트 세트에 적용해보기
print('\n Test Accurary: %.4f' % (model.evaluate(X_test, Y_test)[1]))

# 잘 훈련됐는지 확인해보기
import random
total = len(X_test)
pick = random.sample(range(0, total), 1)
print("{}번 사진".format(pick[0]))

plt.imshow(X_test[pick].reshape(my_size, my_size, 3))

img = X_test[pick].reshape(1, my_size, my_size, 3)
pred = model.predict(img)

# 손상 = 0, 정상 = 1
print('정답:',Y_test[pick])
print('예측', pred)
