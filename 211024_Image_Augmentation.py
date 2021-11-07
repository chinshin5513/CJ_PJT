import timeit
import io
import os
import numpy as np
import tensorflow as tf

from PIL import Image  # 파이썬 이미지 라이브러리
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator

print("텐서플로우 버전 확인:", tf.__version__)
print("케라스 버전 확인:",tf.keras.__version__)

normal_ori_path = "./211024_Photos/n_angle/Original(360)"
normal_crop_path = "./211024_Photos/n_angle/Crop(395)"
normal_ori_list = os.listdir(normal_ori_path)
normal_crop_list = os.listdir(normal_crop_path)
normal_list = normal_ori_list + normal_crop_list
print("정상 대각면 사진 숫자:", len(normal_list))

# 데이터 증가작업에 필요한 라이브러리 불러오기
from numpy import expand_dims
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array
import matplotlib.pyplot as plt

# 가로로 이동하는 함수
def img_aug_width(filename):
    img = load_img(filename)
    data = img_to_array(img)
    img_data = expand_dims(data, 0)
    data_gen = ImageDataGenerator(width_shift_range = 0.1)
    data_iter = data_gen.flow(img_data,
                              batch_size = 2,
                              save_to_dir = "./211024_Photos/n_angle/Augment", 
                              save_prefix = 'width_shift',
                              save_format = 'jpg' )
    batch = data_iter.next()
    image = batch[0].astype('uint16') 

# 세로로 이동하는 함수
def img_aug_height(filename):
    img = load_img(filename)
    data = img_to_array(img)
    img_data = expand_dims(data, 0)
    data_gen = ImageDataGenerator(height_shift_range = 0.1)
    data_iter = data_gen.flow(img_data, 
                              batch_size = 1,
                              save_to_dir = "./211024_Photos/n_angle/Augment", 
                              save_prefix = 'height_shift',
                              save_format = 'jpg' )
    batch = data_iter.next()
    image = batch[0].astype('uint16')

# 회전하는 함수    
def img_aug_rotate(filename):
    img = load_img(filename)
    data = img_to_array(img)
    img_data = expand_dims(data, 0)
    data_gen = ImageDataGenerator(rotation_range = 40)
    data_iter = data_gen.flow(img_data, 
                              batch_size = 1,
                              save_to_dir = "./211024_Photos/n_angle/Augment", 
                              save_prefix = 'rotate',
                              save_format = 'jpg' )
    batch = data_iter.next()
    image = batch[0].astype('uint16')

    
# 이미지 늘리고 줄이는 함수
def img_aug_zoom(filename):
    img = load_img(filename)
    data = img_to_array(img)
    img_data = expand_dims(data, 0)
    data_gen = ImageDataGenerator(zoom_range = [0.6 , 1.4])
    data_iter = data_gen.flow(img_data, 
                              batch_size = 1,
                              save_to_dir = "./211024_Photos/n_angle/Augment", 
                              save_prefix = 'rotate',
                              save_format = 'jpg' )
    batch = data_iter.next()
    image = batch[0].astype('uint16')   
    
    a = range(0, len(normal_list), 2)
for i in a:
    if normal_list[i].startswith("ori"):
        img_aug_width(normal_ori_path + "/" + normal_list[i])
    elif normal_list[i].startswith("crop"):
        img_aug_width(normal_crop_path + "/" + normal_list[i])
    else:
        pass
print("가로 변경 작업: {}회 반복".format(len(a)))
    
b = range(1, len(normal_list), 2)
for i in b:
    if normal_list[i].startswith("ori"):
        img_aug_height(normal_ori_path + "/" + normal_list[i])
    elif normal_list[i].startswith("crop"):
        img_aug_height(normal_crop_path + "/" + normal_list[i])
    else:
        pass
print("세로 변경 작업: {}회 반복".format(len(b)))

c = range(2, len(normal_list), 2)
for i in c:
    if normal_list[i].startswith("ori"):
        img_aug_rotate(normal_ori_path + "/" + normal_list[i])
    elif normal_list[i].startswith("crop"):
        img_aug_rotate(normal_crop_path + "/" + normal_list[i])
    else:
        pass
print("회전 작업: {}회 반복".format(len(c)))


d = range(3, len(normal_list), 2)
for i in d:
    if normal_list[i].startswith("ori"):
        img_aug_rotate(normal_ori_path + "/" + normal_list[i])
    elif normal_list[i].startswith("crop"):
        img_aug_rotate(normal_crop_path + "/" + normal_list[i])
    else:
        pass

print("확대·축소 작업: {}회 반복".format(len(d)))

print("가로 변경 작업: {}회 반복".format(len(a)))
print("세로 변경 작업: {}회 반복".format(len(b)))
print("회전 작업: {}회 반복".format(len(c)))
print("확대·축소 작업: {}회 반복".format(len(d)))
print("-" * 45)
sum_photos = len(a) + len(b) + len(c) + len(d)
print("증가된 사진의 수: {}장".format(sum_photos)) 
print("전체 정상 컨테이너 사진의 수: {}장".format(sum_photos + len(normal_list)))
