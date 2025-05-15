import cv2
import numpy as np

video_path = "pictures/output.avi"                
screenshot_path = "pictures/semenova.png"     
output_dir = "matched_frames"                   

color_ranges = {
    "sun": ((20, 100, 100), (40, 255, 255)),       # солнце
    "star": ((0, 150, 100), (10, 255, 255))        # звезда на елке 
}


reference_masks = {}
hsv_ref = cv2.cvtColor(cv2.imread(screenshot_path), cv2.COLOR_BGR2HSV)  

# бинарные маски по каждому цветовому диапазону
for name, (lower, upper) in color_ranges.items():
    reference_masks[name] = cv2.inRange(hsv_ref, np.array(lower), np.array(upper))  

# насколько хорошо две маски пересекаются
def iou(mask1, mask2):
    # подводим к одному размеру
    if mask1.shape != mask2.shape:
        mask1 = cv2.resize(mask1, (mask2.shape[1], mask2.shape[0]), interpolation=cv2.INTER_NEAREST)

    # пересечение и объединение
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    return inter / union if union > 0 else 0

cap = cv2.VideoCapture(video_path)                         
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))       # общ кол-во кадров 

matched_frames = 0      # счётчик совпавших кадров
frame_idx = 0           # индекс текущего кадра

while True:
    ret, frame = cap.read()   
    if not ret:                
        break

    # кадр в HSV для чтобы выделить маски по цвету
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    frame_masks = {
        name: cv2.inRange(hsv, np.array(lower), np.array(upper))
        for name, (lower, upper) in color_ranges.items()
    }

    # сравнение масок с тем которые мы сделали 
    matches = sum(iou(reference_masks[n], frame_masks[n]) >= 0.2 for n in reference_masks)

    if matches >= 2:
        matched_frames += 1
        cv2.imwrite(f"{output_dir}/frame_{frame_idx:04d}.jpg", frame)

    frame_idx += 1     

cap.release()

print(f"мое изображение встретилось {matched_frames} раз") # !!!! в папке matched_frames сохранились все разы, когда встречалось мое изображение