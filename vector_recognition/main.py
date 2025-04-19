import numpy as np
from skimage.measure import regionprops, label
import matplotlib.pyplot as plt

def extractor(region):
    normalized_area = region.area / region.image.size
    cy_rel, cx_rel = region.centroid_local
    cy_rel /= region.image.shape[0]
    cx_rel /= region.image.shape[1]
    normalized_perimeter = region.perimeter / region.image.size
    eccentricity = region.eccentricity
    holes = 1 - region.euler_number
    solidity = region.solidity
    aspect_ratio = region.image.shape[1] / region.image.shape[0]

    cy_idx = int(region.centroid_local[0])
    cx_idx = int(region.centroid_local[1])
    row_crossings = np.sum(region.image[cy_idx, :-1] != region.image[cy_idx, 1:])
    col_crossings = np.sum(region.image[:-1, cx_idx] != region.image[1:, cx_idx])

    h, w = region.image.shape[0] // 2, region.image.shape[1] // 2
    sym_h = np.sum(region.image[:h, :] == np.flipud(region.image[-h:, :])) / (h * region.image.shape[1] * 2)
    sym_v = np.sum(region.image[:, :w] == np.fliplr(region.image[:, -w:])) / (w * region.image.shape[0] * 2)

    return np.array([
        normalized_area, cy_rel, cx_rel, normalized_perimeter, eccentricity,
        holes, solidity, aspect_ratio, row_crossings, col_crossings,
        sym_v, sym_h
    ])

def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

def classify_nearest_neighbor(feature_vector, templates):
    result_label = "_"
    min_dist = float('inf')
    for key, template_vector in templates.items():
        d = euclidean_distance(feature_vector, template_vector)
        if d < min_dist:
            result_label = key
            min_dist = d
    return result_label

alphabet_large = plt.imread("vector_recognition/alphabet.png")[:, :, :3]
binary_large = alphabet_large.mean(axis=2) > 0.5
regions_large = regionprops(label(binary_large))

alphabet_small = plt.imread("vector_recognition/alphabet-small.png")[:, :, :3]
binary_small = alphabet_small.mean(axis=2) < 0.5
regions_small = regionprops(label(binary_small))

templates = {}
template_symbols = ["8", "0", "A", "B", "1", "W", "X", "*", "/", "-"]
for i, symbol in enumerate(template_symbols[:len(regions_small)]):
    templates[symbol] = extractor(regions_small[i])

results = {}
for region in regions_large:
    if region.area < 10:
        continue
    v = extractor(region)
    symbol = classify_nearest_neighbor(v, templates)
    results[symbol] = results.get(symbol, 0) + 1

print("Результат классификацит:")
print(results)