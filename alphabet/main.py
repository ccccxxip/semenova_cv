import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops, label
from skimage.morphology import binary_dilation
from pathlib import Path


def count_holes_in_symbol(symbol_region):
    symbol_shape = symbol_region.image.shape
    padded_image = np.zeros((symbol_shape[0] + 2, symbol_shape[1] + 2))
    padded_image[1:-1, 1:-1] = symbol_region.image
    
    # отверстие==объект
    inverted_image = np.logical_not(padded_image)
    
    # отверстие
    labeled_holes = label(inverted_image)
    
    return np.max(labeled_holes) - 1


def count_vertical_lines(symbol_region):
    """количество вертикальных линий в символе"""
    return np.all(symbol_region.image, axis=0).sum()


def has_more_lines_on_left(symbol_region):
    """больше вертикальных линий в левой половине символа?"""
    vertical_profiles = symbol_region.image.mean(axis=0) == 1
    half_width = len(vertical_profiles) // 2
    return np.sum(vertical_profiles[:half_width]) > np.sum(vertical_profiles[half_width:])


def recognize_symbol(symbol_region):
    if np.all(symbol_region.image):
        return "-"
    
    holes_count = count_holes_in_symbol(symbol_region)
    
    if holes_count == 2:
        vertical_lines_count = count_vertical_lines(symbol_region)
        left_heavy = has_more_lines_on_left(symbol_region)
        
        
        centroid_y, centroid_x = symbol_region.centroid_local
        normalized_centroid_x = centroid_x / symbol_region.image.shape[1]
        
        if left_heavy and normalized_centroid_x < 0.45:
            return "B"
        return "8"
    
    elif holes_count == 1:
        centroid_y, centroid_x = symbol_region.centroid_local
        normalized_centroid_x = centroid_x / symbol_region.image.shape[1]
        normalized_centroid_y = centroid_y / symbol_region.image.shape[0]
        
        if has_more_lines_on_left(symbol_region):
            # D и P имеют больше линий слева
            if normalized_centroid_x > 0.45 or normalized_centroid_y > 0.45:
                return "D"
            else:
                return "P"
        
        # Ноль симметричен
        if abs(normalized_centroid_x - normalized_centroid_y) < 0.05:
            return "0"
        return "A"
    
    else:
        if count_vertical_lines(symbol_region) >= 3:
            return "1"
        
        if symbol_region.eccentricity <= 0.45:
            return "*"
        
        # Анализ символов без отверстий
        inverted_image = ~symbol_region.image
        dilated_image = binary_dilation(inverted_image, np.ones((3, 3)))
        labeled_components = label(dilated_image)
        
        components_count = np.max(labeled_components)
        if components_count == 2:
            return "/"
        elif components_count == 4:
            return "X"
        else:
            return "W"
    
    return "#" 


symbols_image = plt.imread(Path(__file__).parent / "symbols.png")
gray_image = symbols_image[:, :, :-1].mean(axis=2) 
binary_image = gray_image > 0  # Бинаризация

labeled_symbols = label(binary_image)
symbol_regions = regionprops(labeled_symbols)

recognition_results = {}

output_directory = Path(__file__).parent / "out"
output_directory.mkdir(exist_ok=True)

plt.figure()
for idx, region in enumerate(symbol_regions):
    print(f"{idx + 1}/{len(symbol_regions)}")
    
    recognized_symbol = recognize_symbol(region)
    
    if recognized_symbol not in recognition_results:
        recognition_results[recognized_symbol] = 0
    recognition_results[recognized_symbol] += 1
    
    plt.cla()
    plt.title(recognized_symbol)
    plt.imshow(region.image)
    plt.savefig(output_directory / f"symbol_{idx:03d}.png")

print("Результаты :")
print(recognition_results)