import cv2
import numpy as np

cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
capture.set(cv2.CAP_PROP_EXPOSURE, -3)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'deal_with_it\haarcascade-frontalface-default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'deal_with_it\haarcascade-eye.xml')

# Загружаем изображение очков (черные на белом фоне)
glasses_img = cv2.imread("deal_with_it\deal-with-it.png", cv2.IMREAD_COLOR)  # Ч/Б или цветное

if glasses_img is None:
    print("Ошибка: Не удалось загрузить glasses.png")
    exit()

# Преобразуем белый фон в прозрачный (альфа-канал)
# 1. Создаем маску: белый фон = 0, остальное = 255
gray_glasses = cv2.cvtColor(glasses_img, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray_glasses, 220, 255, cv2.THRESH_BINARY_INV)  # Порог для белого фона
mask_inv = cv2.bitwise_not(mask)

# 2. Извлекаем только очки (черные части)
glasses_fg = cv2.bitwise_and(glasses_img, glasses_img, mask=mask)

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break
    
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    # Обнаруживаем лица
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        # Область лица для поиска глаз
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Обнаруживаем глаза в области лица
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
        
        # Если найдено ровно два глаза
        if len(eyes) == 2:
            # Сортируем глаза по координате x (левый и правый)
            eyes = sorted(eyes, key=lambda e: e[0])
            
            # Координаты левого и правого глаза
            (ex1, ey1, ew1, eh1) = eyes[0]
            (ex2, ey2, ew2, eh2) = eyes[1]
            
            # Вычисляем ширину и высоту очков
            glasses_width = int((ex2 + ew2) - ex1)
            glasses_height = int(max(eh1, eh2) * 1.5)
            
            # Позиция очков (немного выше глаз)
            glasses_y = y + ey1 - glasses_height // 3
            glasses_x = x + ex1
            
            # Масштабируем очки
            resized_glasses = cv2.resize(glasses_fg, (glasses_width, glasses_height))
            resized_mask = cv2.resize(mask, (glasses_width, glasses_height))
            resized_mask_inv = cv2.resize(mask_inv, (glasses_width, glasses_height))
            
            # Область, куда будем накладывать очки
            roi = frame[glasses_y:glasses_y + glasses_height, glasses_x:glasses_x + glasses_width]
            
            # Удаляем фон (белый) и накладываем очки
            roi_bg = cv2.bitwise_and(roi, roi, mask=resized_mask_inv)
            roi_fg = cv2.bitwise_and(resized_glasses, resized_glasses, mask=resized_mask)
            dst = cv2.add(roi_bg, roi_fg)
            
            # Вставляем обратно в кадр
            frame[glasses_y:glasses_y + glasses_height, glasses_x:glasses_x + glasses_width] = dst
    
    # Отображаем результат
    cv2.imshow("Camera", frame)
    
    # Выход по нажатию 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
capture.release()
cv2.destroyAllWindows()