import numpy as np  
import matplotlib.pyplot as plt  
from skimage.measure import label, regionprops   
from skimage.filters import sobel, threshold_otsu 
from skimage.color import rgb2gray 
from skimage.morphology import binary_erosion, binary_dilation  
from pathlib import Path 

image_path = Path(__file__).parent / 'images'
total_pen = 0  # счетчик карандашей

for img_file in image_path.glob('*.jpg'):

    image = plt.imread(img_file)
    
    # градация серого обработка
    gray = rgb2gray(image)
    
    edges = sobel(gray)
    
    # подбор порога
    thresh = threshold_otsu(edges)
    
    # бинаризация 
    binary = edges >= thresh
    
    processed = binary_dilation(
                  binary_erosion(
                    binary_dilation(binary, np.ones((10,10))), 
                    np.ones((3,3))
                  ), 
                  np.ones((10,10))
                )
    
    # маркировка
    labeled = label(processed)
    
    # свойства региона
    regions = regionprops(labeled)
    
    # min S
    min_area = 0.005 * gray.size
    
    # отбор карандашей 
    pencils = [r for r in regions 
              if 10 <= (r.major_axis_length / r.minor_axis_length) <= 20 
              and r.area > min_area]
    # кол-во карандашей на изображении №
    print(f"кол-во карандашей на {img_file.name}: {len(pencils)}")
    
    total_pen += len(pencils)
    
print(f"всего карандашей: {total_pen}")