import numpy as np
import matplotlib.pyplot as plt

X = np.load('./211024_rgb_cnn_input.npy')
Y = np.load('./211024_rgb_cnn_target.npy')

# 인풋과 타겟의 배열 살펴보기
print("인풋 세트의 배열형태: ", X.shape)
print("타겟 세트의 배열형태: ", Y.shape)

# 각 사진의 비율 알아보기
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

# 데이터 내에서 훈련과 테스트 세트 나누기(비율=0.2) 
from sklearn.model_selection import train_test_split
# 데이터의 셔플을 진행하지 않으면 편향될 수 있으므로 셔플을 실시함
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, shuffle =true, random_state=42)

# 255로 나눠주어 전처리 시작(사진 데이터는 0~255까지 척도가 정해져 있으므로, 나누어 전처리해주어야 함)
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

# 딥러닝 모델 구현하기 : 컨볼루션 필터 64개, 필터 사이즈 3×3, 맥스 풀링 2, 완전연결층 256개, 드롭아웃 25%
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(128, 128, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid')) 
model.summary()

# 모델 컴파일하기
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 훈련하기 : 학습횟수 50회, 1번에 들어가는 데이터의 숫자 64개
history = model.fit(X_train, Y_train, epochs=50, batch_size=64)

# 사진 하나 추출하여 예측이 잘 되었는지 확인하기
import random
total = len(X_test)
pick = random.sample(range(0, total), 1)
print(pick[0],'번 사진')
plt.imshow(X_test[pick].reshape(my_size, my_size, 3))
img = X_test[pick].reshape(1, my_size, my_size, 3)
pred = model.predict(img)
# 손상 = 0, 정상 = 1
print('정답:',Y_test[pick])
print('예측', pred)

# 테스트 세트에 적용해보기
print('\n Test Accurary: %.4f' % (model.evaluate(X_test, Y_test)[1]))

# ------------<교차검증 실시하기>------------

# 교차검증 실행하기 위해서 모델을 다시 정의하자.
model2 = Sequential()
model2.add(Conv2D(64, kernel_size=(3, 3), input_shape=(128, 128, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.25))
model2.add(Flatten())
model2.add(Dense(256, activation='relu'))
model2.add(Dropout(0.25))
model2.add(Dense(1, activation='sigmoid')) 
model2.summary()

# 모델 컴파일하기
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 훈련하기 : 학습횟수 50회, 1번에 들어가는 데이터의 숫자 64개
history2 = model2.fit(X_train, Y_train, validation_data=(X_test, Y_test),  epochs=50, batch_size=64)

# 테스트 세트에 적용해보기
print('\n Model2 Test Accurary: %.4f' % (model2.evaluate(X_test, Y_test)[1]))

y_vloss = history2.history['val_loss']  # 테스트 세트의 오차
y_loss = history2.history['loss']  #  학습 세트의 오차

# 훈련 셋과 테스트 셋의 오차 그림으로 
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c='red', label='Test_Set_Loss')
plt.plot(x_len, y_loss, marker='^', c='green', label='Train_Set_Loss')
plt.legend(loc='upper right')
# plt.axis([0, 20, 0, 0.35])
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
