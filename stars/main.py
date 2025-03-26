import numpy as np
import matplotlib.pyplot as plt

img = np.load("stars\stars.npy")

if len(img.shape) == 3:
    img = np.mean(img, axis=2)

stars = img > 0.5

plus = np.array([    # плюсик
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
])

cross = np.array([    # крестик
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
])

num_plus = 0
num_cross = 0

for row in range(stars.shape[0] - 2):
    for col in range(stars.shape[1] - 2):
        part = stars[row:row+3, col:col+3]
        
        if np.array_equal(part, plus):
            num_plus += 1
        
        if np.array_equal(part, cross):
            num_cross += 1

print(f"плюсов: {num_plus}")
print(f"крестов: {num_cross}")
print(f"всего: {num_plus + num_cross}")

plt.imshow(img)
plt.show()