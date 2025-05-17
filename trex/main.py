import cv2
import numpy as np
import pyautogui
import time
import mss

# область экрана, где расположена игра 
GAME_REGION = {'top': 262, 'left': 1030, 'width': 789, 'height': 17+5}


NEAR_OFFSET_BASE = 170     #  смещение зоны от динозавра
NEAR_WIDTH = 67        # ширина ближней зоны
NEAR_HEIGHT = 40             # высота ближней зоны 
NEAR_PIXEL_THRESHOLD = 120   # кол-во тёмных пикселей, при котором считается, что есть препятствие

# дальняя зона
FAR_OFFSET_BASE = 220 
FAR_WIDTH = 90 
FAR_HEIGHT = 40 
FAR_PIXEL_THRESHOLD = 250
 
#  время между прыжками 
JUMP_COOLDOWN = 0.45
 
#  коэф для сложности 
OFFSET_SPEED_COEFF = 1.5          # быстрота сдвига зоны 
PIXEL_THRESHOLD_COEFF = 1.0      
 
#   смещения зон с учётом текущего счёта
def get_dynamic_offset(score, base_offset):
    offset = base_offset - int(score * OFFSET_SPEED_COEFF)
    if offset < 100:  # мин смещение
        offset = 100
    return offset
 
#  обнаружение препятствий 
def detect_obstacle(img, threshold):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                     # чб
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    dark_pixels = cv2.countNonZero(binary)
    return dark_pixels > threshold, dark_pixels
 
#  прыжок с приседом
def jump_with_duck():
    pyautogui.press('space')    # прыжок
    time.sleep(0.15)            # задержка — время, пока динозавр в воздухе
    pyautogui.keyDown('down')   # приседание (ускоряет приземление)
    time.sleep(0.1)             # держим вниз
    pyautogui.keyUp('down')     # отпускаем вниз
 
def main():
    sct = mss.mss()
    time.sleep(2) 
    print("Игра началась!")
    
    score = 0
    last_jump_time = 0      # время последнего прыжка

    cv2.namedWindow("window")

    while True:
        full_img = np.array(sct.grab(GAME_REGION))

        near_offset = get_dynamic_offset(score, NEAR_OFFSET_BASE)
        far_offset = get_dynamic_offset(score, FAR_OFFSET_BASE)
        # коорд ближн зоны 
        near_x1 = near_offset
        near_y1 = 105
        near_x2 = near_x1 + NEAR_WIDTH
        near_y2 = near_y1 + NEAR_HEIGHT
        # коорд дальн зоны 
        far_x1 = far_offset
        far_y1 = 105
        far_x2 = far_x1 + FAR_WIDTH
        far_y2 = far_y1 + FAR_HEIGHT

        near_zone = full_img[near_y1:near_y2, near_x1:near_x2]
        far_zone = full_img[far_y1:far_y2, far_x1:far_x2]

        near_threshold = int(NEAR_PIXEL_THRESHOLD * PIXEL_THRESHOLD_COEFF)
        far_threshold = int(FAR_PIXEL_THRESHOLD * PIXEL_THRESHOLD_COEFF)

        # проверка на препятствия
        near_detected, _ = detect_obstacle(near_zone, near_threshold)
        far_detected, _ = detect_obstacle(far_zone, far_threshold)

        current_time = time.time()

        if near_detected and (current_time - last_jump_time) > JUMP_COOLDOWN:
            jump_with_duck()
            last_jump_time = current_time

        elif not near_detected and far_detected and (current_time - last_jump_time) > JUMP_COOLDOWN:
            jump_with_duck()
            last_jump_time = current_time

            #cv2.rectangle(full_img, (near_x1, near_y1), (near_x2, near_y2), (255, 0, 0), 2)
            #cv2.rectangle(full_img, (far_x1, far_y1), (far_x2, far_y2), (0, 255, 0), 2)

            #cv2.imshow('Game', full_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        score += 1
        time.sleep(0.01)  # задержка между прыжками

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
