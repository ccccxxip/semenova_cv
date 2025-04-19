import numpy as np
from skimage.measure import regionprops, label
import matplotlib.pyplot as plt

def extractor(region):
    
    # отношение площади к размеру изображения о
    area_n = region.area / region.image.size
    
    cy, cx = region.centroid_local
    cy /= region.image.shape[0]  # Нормировка по высоте
    cx /= region.image.shape[1]  # Нормировка по ширине
    
    # периметр
    perim_n = region.perimeter / region.image.size
    
    # вытянутость фигуры
    ecc = region.eccentricity
    
    # Количество дырок
    holes = 1 - region.euler_number
    
    # отношение площади к площади выпуклой оболочки
    sol = region.solidity
    
    # ширина/высота
    ar = region.image.shape[1] / region.image.shape[0]

    y = int(cy * region.image.shape[0])  # координата y центра
    x = int(cx * region.image.shape[1])  # координата x центра
    
    row_cross = np.sum(region.image[y, :-1] != region.image[y, 1:])
    
    col_cross = np.sum(region.image[:-1, x] != region.image[1:, x])

    # Проверка симметрии:
    h, w = region.image.shape[0] // 2, region.image.shape[1] // 2  # Половины размеров
    
    # сравнение левой и правой половин
    sym_v = np.sum(region.image[:h, :] == np.flipud(region.image[-h:, :])) / (h * region.image.shape[1] * 2)
    
    # сравнение верхней и нижней половин
    sym_h = np.sum(region.image[:, :w] == np.fliplr(region.image[:, -w:])) / (w * region.image.shape[0] * 2)

    return np.array([
        area_n, cy, cx, perim_n, ecc,
        holes, sol, ar, row_cross, col_cross,
        sym_v, sym_h
    ])

def euclidean_distance(v1, v2):
    """Вычисляет евклидово расстояние между двумя векторами признаков"""
    return np.sqrt(np.sum((v1 - v2) ** 2))

def classify_nearest_neighbor(feature_vector, templates):
   
    result_label = "_"  # дефолтная метка 
    min_dist = float('inf')  # Начальное минимальное расстояние
    
    for key, template_vector in templates.items():
        d = euclidean_distance(feature_vector, template_vector)
        if d < min_dist:
            result_label = key  # обновленная меткф
            min_dist = d       # обновленное минимальное расстояние
    
    return result_label

alphabet_large = plt.imread("vector_recognition/alphabet.png")[:, :, :3]  
binary_large = alphabet_large.mean(axis=2) > 0.5  # Бинаризация
regions_large = regionprops(label(binary_large))  

alphabet_small = plt.imread("vector_recognition/alphabet-small.png")[:, :, :3]
binary_small = alphabet_small.mean(axis=2) < 0.5  # Бинаризация 
regions_small = regionprops(label(binary_small)) 

# вектор признаков
templates = {}
template_symbols = ["8", "0", "A", "B", "1", "W", "X", "*", "/", "-"]
for i, symbol in enumerate(template_symbols[:len(regions_small)]):
    templates[symbol] = extractor(regions_small[i])  

results = {}
for region in regions_large:
    if region.area < 10:  # шум
        continue
    v = extractor(region)  
    symbol = classify_nearest_neighbor(v, templates) 
    results[symbol] = results.get(symbol, 0) + 1 

print("Результат классификациц: ")
print(results)