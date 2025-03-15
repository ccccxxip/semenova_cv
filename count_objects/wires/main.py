import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.morphology import binary_erosion

data = np.load("wires6npy.txt")

labeled = label(data) # маркировка изображения 

num_wires = np.max(labeled)
print(f"количество проводов: {num_wires}")

for wire_id in range(1, num_wires + 1):
    wire_mask = labeled == wire_id # маска текущего провода 
    result = binary_erosion(wire_mask, np.ones((3, 1))) # разделение на части 
    parts_labeled = label(result)

    num_parts = np.max(parts_labeled)
    if num_parts == 0:
        print("провод не сущ")
    else:
        print(f"провод {wire_id} разделен на {num_parts} частей")

    plt.figure()
    plt.title(f"провод {wire_id} (частей: {num_parts})")
    plt.imshow(parts_labeled)
    plt.show()