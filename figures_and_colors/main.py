from skimage.color import rgb2hsv 
import matplotlib.pyplot as plt 
from skimage.measure import label, regionprops 

def get_color(hue_value):
    color_ranges = {
        "красный": (0.0, 0.19202898),     
        "оранжевый": (0.19202898, 0.30476192), 
        "желтый": (0.30476192, 0.41509435),    
        "зеленый": (0.41509435, 0.60897434),   
        "голубой": (0.60897434, 0.8333333),    
        "розовый": (0.8333333, 1.0)
    }
    
    # поиск подходящего диапазона для заданного значения Hue
    for color, (lower, upper) in color_ranges.items():
        if lower <= hue_value < upper:
            return color

image = plt.imread("figures_and_colors/balls_and_rects.png")

binary = image.mean(axis=2) > 0  # бинаризация 

labeled = label(binary) # маркируем

regions = regionprops(labeled)

balls = {}  # шарики по цветам
rects = {}  # прямоугольники по цветам 

for region in regions:
    # координаты центра области 
    y, x = region.centroid
    
    hue = rgb2hsv(image[int(y), int(x)])[0] # определеник цвета
    
    color = get_color(hue)
    
    # тип фигуры 
    target_dict = balls if region.eccentricity == 0 else rects
    
    # get() возвр. 0 если цвета нет в словаре
    target_dict[color] = target_dict.get(color, 0) + 1

# обще количество фигур
total = sum(balls.values()) + sum(rects.values())

print("всего фигур:", total) 
print("шарики:", balls)      
print("прямоугольники:", rects)  