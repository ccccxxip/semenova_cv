#разработать программу для подсчёта количества отверстий на цифровом бинарном изображении

import numpy as np
import matplotlib.pyplot as plt

import os
os.chdir(r"C:\Users\Компьютер\OneDrive\files")  # путь к нужному каталогу



external = np.array([([0, 0], [0, 1]),
               ([0, 0], [1, 0]),
              ([0, 1], [0, 0]),
              ( [1, 0], [0, 0])])     # внешние и внутренние углы
internal = np.logical_not(external)
cross = np.array([([1, 0], [0, 1]),
               ([0, 1], [1, 0])]) # пересечения 


def match(fragment, masks):
    for mask in masks:
        if np.all((mask != 0) == (fragment != 0)):  
            return True
    return False


def count_obj(image):
    E = 0
    for y in range (0, image.shape[0]-1):
        for x in range(0, image.shape[1]-1):
            sub = image[y:y+2, x:x+2]
            if match(sub, external):
                E += 1
            elif match(sub, internal):
                E -= 1
            elif match(sub, cross):
                E += 2
    return E/ 4 


image_first = np.load("example1.npy")
image_second = np.load("example2.npy")

print(image_first.shape)
print(image_second.shape)

plt.figure(figsize=(12, 12))
plt.subplot(2, 4, 1)
plt.title("image first")
plt.imshow(image_first)
plt.subplot(2, 4, 2)
plt.title("image second: side 1")
plt.imshow(image_second[:, :, 0])
plt.subplot(2, 4, 3)
plt.title("image second: side 2")
plt.imshow(image_second[:, :, 1])
plt.subplot(2, 4, 4)
plt.title("image second: side 3")
plt.imshow(image_second[:, :, 2])

print("count of objects in the first image: ", count_obj(image_first))

sides_of_second_image = [count_obj(image_second[:, :, 0]), 
                         count_obj(image_second[:, :, 1]), 
                         count_obj(image_second[:, :, 2])]
print("count of objects in the second image: ", sum(sides_of_second_image))


plt.show()