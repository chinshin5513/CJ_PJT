import timeit
import io
import os
import glob
import numpy as np
import tensorflow as tf

from PIL import Image  # 파이썬 이미지 라이브러리
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


print("텐서플로우 버전 확인:", tf.__version__)
print("케라스 버전 확인:",tf.keras.__version__)

# 정상 컨테이너 경로 지정
normal_ori_path = "./211024_Photos/n_angle/Original(360)"
normal_crop_path = "./211024_Photos/n_angle/Crop(395)"
normal_augment_path = "./211024_Photos/n_angle/Augment"

# 손상 컨테이너 경로 지정
damaged_ori_path = "./211024_Photos/d_angle/Original(163)"
damaged_crop_path = "./211024_Photos/d_angle/Crop(109)"
damaged_augment_path = "./211024_Photos/d_angle/Augment(2000)"

# 손상 컨테이너 사진 숫자 구하기
damaged_list = os.listdir(damaged_ori_path) + os.listdir(damaged_crop_path) + os.listdir(damaged_augment_path)
print("손상 대각면 사진 숫자:", len(damaged_list))  # 

# 정상 컨테이너 사진 숫자 구하기
normal_list = os.listdir(normal_ori_path) + os.listdir(normal_crop_path) + os.listdir(normal_augment_path)
print("정상 대각면 사진 숫자:", len(normal_list))

x, y = len(damaged_list), len(normal_list)
total = x+y
d_ratio = len(damaged_list)/total*100
n_ratio = len(normal_list)/total*100

print("손상 사진 비율: {0:.2f}%, 정상 사진 비율: {1:.2f}%".format(d_ratio, n_ratio))

my_size = 128  # 각종 파라미터 조정하기 위해서...(AlexNet = 227, VGG16 = 224)

# 손상 컨테이너 이름 가져오기
d_photo_list = []
for f in os.listdir(damaged_ori_path):
    if 'jpg' in f:
        d_photo_list.append(f)
    elif 'JPG' in f:
        d_photo_list.append(f)  
    elif 'jfif' in f:
        d_photo_list.append(f)    
    elif 'png' in f:
        d_photo_list.append(f)
    elif 'PNG' in f:
        d_photo_list.append(f)
    else:
        pass
    
for f in os.listdir(damaged_crop_path):
    if 'jpg' in f:
        d_photo_list.append(f)
    elif 'JPG' in f:
        d_photo_list.append(f)  
    elif 'jfif' in f:
        d_photo_list.append(f)    
    elif 'png' in f:
        d_photo_list.append(f)
    elif 'PNG' in f:
        d_photo_list.append(f)
    else:
        pass

for f in os.listdir(damaged_augment_path):
    if 'jpg' in f:
        d_photo_list.append(f)
    elif 'JPG' in f:
        d_photo_list.append(f)  
    elif 'jfif' in f:
        d_photo_list.append(f)    
    elif 'png' in f:
        d_photo_list.append(f)
    elif 'PNG' in f:
        d_photo_list.append(f)
    else:
        pass       
print("손상 컨테이너 사진: {}장".format(len(d_photo_list))) 

def image_to_numpy(filename, my_size=128):
    img = Image.open(filename)
    img = img.convert("RGB")
    img = img.resize((my_size, my_size))
    pixel_data = img.getdata()
    pixels = np.array(pixel_data)
    return pixels

d_img_array = []
try:
    for i in range(len(d_photo_list)):
        x = d_photo_list[i]
        if x.startswith("ori"): 
            img = damaged_ori_path + '/' + x
            y = image_to_numpy(img)
            d_img_array.append(y)  # 오리지널 데이터(Labelling)는 "d_con_000.확장자"로 동일한 규칙을 가짐
        elif x.startswith("crop"):
            img = damaged_crop_path + '/' + x
            y = image_to_numpy(img)
            d_img_array.append(y) 
        else:
            img = damaged_augment_path + '/' + x
            y = image_to_numpy(img)
            d_img_array.append(y)  # 증강 데이터는 파일명에 규칙이 없음   
    # 리스트를 넘파이 배열로 변경하기
    d_img_array = np.array(d_img_array) 
    print("손상 컨테이너 차원:", d_img_array.shape)
    print("픽셀값 → Numpy 정상 변환 완료!")
except:
    print("바_______보")  
    
print("DB(행): ", len(d_img_array[0]))
print("DB(열): ", len(d_img_array))
print("데이터 타입: ", type(d_img_array))
print("DB(전체): ", d_img_array.shape)

damage_class = np.zeros((len(d_img_array), ))
print("타겟 데이터 타입: ", type(damage_class))
print("타겟 데이터 차원: ", damage_class.shape)
print("타겟 데이터: {}개".format(damage_class.shape))

'''
# 정우가 만들어 준 깔끔한 코드(글로브 함수로 받는 방법, 추후 테스트 )

import os
import glob
from PIL import Image

# 정상 컨테이너 경로 지정
normal_ori_path = glob.glob("./211024_Photos/n_angle/Original(360)/*")
normal_crop_path = glob.glob("./211024_Photos/n_angle/Crop(395)")
normal_augment_path = glob.glob("./211024_Photos/n_angle/Augment")

# 손상 컨테이너 경로 지정
damaged_ori_path = glob.glob("./211024_Photos/d_angle/Original(163)")
damaged_crop_path = glob.glob("./211024_Photos/d_angle/Crop(109)")
damaged_augment_path = glob.glob("./211024_Photos/d_angle/Augment(2000)")


d_photo_list = []
for img_path in normal_ori_path:
    title, ext = os.path.splitext(img_path)
    if ext in ['.jpg','.jfif','.JPG', '.png', 'PNG']:
        d_photo_list.append(img_path)
    else:
        print("멍충아!!!!!")
    
print(len(d_photo_list))

'''

# 손상 컨테이너 이름 가져오기
n_photo_list = []
for f in os.listdir(normal_ori_path):
    if 'jpg' in f:
        n_photo_list.append(f)
    elif 'JPG' in f:
        n_photo_list.append(f)  
    elif 'jfif' in f:
        n_photo_list.append(f)    
    elif 'png' in f:
        n_photo_list.append(f)
    elif 'PNG' in f:
        n_photo_list.append(f)
    else:
        pass
    
for f in os.listdir(normal_crop_path):
    if 'jpg' in f:
        n_photo_list.append(f)
    elif 'JPG' in f:
        n_photo_list.append(f)  
    elif 'jfif' in f:
        n_photo_list.append(f)    
    elif 'png' in f:
        n_photo_list.append(f)
    elif 'PNG' in f:
        n_photo_list.append(f)
    else:
        pass

for f in os.listdir(normal_augment_path):
    if 'jpg' in f:
        n_photo_list.append(f)
    elif 'JPG' in f:
        n_photo_list.append(f)  
    elif 'jfif' in f:
        n_photo_list.append(f)    
    elif 'png' in f:
        n_photo_list.append(f)
    elif 'PNG' in f:
        n_photo_list.append(f)
    else:
        pass       
print("정상 컨테이너 사진: {}장".format(len(n_photo_list))) 

n_img_array = []
try:
    for i in range(len(n_photo_list)):
        x = n_photo_list[i]
        if x.startswith("ori"): 
            img = normal_ori_path + '/' + x
            y = image_to_numpy(img)
            n_img_array.append(y)  # 오리지널 데이터(Labelling)는 "n_con_000.확장자"로 동일한 규칙을 가짐
        elif x.startswith("crop"):
            img = normal_crop_path + '/' + x
            y = image_to_numpy(img)
            n_img_array.append(y) 
        else:
            img = normal_augment_path + '/' + x
            y = image_to_numpy(img)
            n_img_array.append(y)  # 증강 데이터는 파일명에 규칙이 없음   
    # 리스트를 넘파이 배열로 변경하기
    n_img_array = np.array(n_img_array) 
    print("정상 컨테이너 차원:", n_img_array.shape)
    print("픽셀값 → Numpy 정상 변환 완료!")
except:
    print("바_______보")
    
total = len(d_img_array) + len(n_img_array)
print("전체 사진 갯수: {}장".format(total))

print("\n------input information------")
# 인풋 세트 만들기
input = np.append(d_img_array, n_img_array, axis = 0)
input = input.reshape(total, my_size, my_size, 3)
print("인풋 세트 배열의 형태: ", input.shape)


print("\n------target information------")
# 정상 컨테이너 타겟 만들어 주기
normal_class = np.ones((len(n_img_array), ))
print("정상 타겟 데이터의 배열:", normal_class.shape)

# 손상 컨테이터 타겟 만들어주기
damage_class = np.zeros((len(d_img_array), ))
print("손상 타겟 데이터의 배열:", damage_class.shape)

# 세트 만들기
target = np.append(damage_class, normal_class, axis=0)
print("타겟 세트 배열의 형태: ", target.shape)

# 인풋과 타겟 NumPy 파일로 저장하기
np.save('./211024_rgb_cnn_input.npy', input)
np.save('./211024_rgb_cnn_target.npy', target)

