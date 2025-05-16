import cv2
import numpy as np
import pyautogui
import time
import mss

GAME_REGION = {'top': 262, 'left': 1032, 'width': 789, 'height': 175}

NEAR_OFFSET_BASE = 170
NEAR_WIDTH = 65
NEAR_HEIGHT = 40
NEAR_PIXEL_THRESHOLD = 120

FAR_OFFSET_BASE = 220
FAR_WIDTH = 90
FAR_HEIGHT = 40
FAR_PIXEL_THRESHOLD = 250

JUMP_COOLDOWN = 0.45

OFFSET_SPEED_COEFF = 1.5
PIXEL_THRESHOLD_COEFF = 1.0

def get_dynamic_offset(score, base_offset):
    offset = base_offset - int(score * OFFSET_SPEED_COEFF)
    if offset < 100:
        offset = 100
    return offset

def detect_obstacle(img, threshold):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    dark_pixels = cv2.countNonZero(binary)
    return dark_pixels > threshold, dark_pixels

def jump_with_duck():
    pyautogui.press('space')    # Прыжок
    time.sleep(0.15)            # Небольшая задержка в воздухе
    pyautogui.keyDown('down')   # Опускаемся вниз (приседаем)
    time.sleep(0.1)             # Держим приседание
    pyautogui.keyUp('down')     # Отпускаем кнопку

def main():
    sct = mss.mss()
    time.sleep(2)
    print("Игра началась!")

    score = 0
    last_jump_time = 0

    while True:
        full_img = np.array(sct.grab(GAME_REGION))

        near_offset = get_dynamic_offset(score, NEAR_OFFSET_BASE)
        far_offset = get_dynamic_offset(score, FAR_OFFSET_BASE)

        near_x1 = near_offset
        near_y1 = 105
        near_x2 = near_x1 + NEAR_WIDTH
        near_y2 = near_y1 + NEAR_HEIGHT

        far_x1 = far_offset
        far_y1 = 105
        far_x2 = far_x1 + FAR_WIDTH
        far_y2 = far_y1 + FAR_HEIGHT

        near_zone = full_img[near_y1:near_y2, near_x1:near_x2]
        far_zone = full_img[far_y1:far_y2, far_x1:far_x2]

        near_threshold = int(NEAR_PIXEL_THRESHOLD * PIXEL_THRESHOLD_COEFF)
        far_threshold = int(FAR_PIXEL_THRESHOLD * PIXEL_THRESHOLD_COEFF)

        near_detected, _ = detect_obstacle(near_zone, near_threshold)
        far_detected, _ = detect_obstacle(far_zone, far_threshold)

        current_time = time.time()

        if near_detected and (current_time - last_jump_time) > JUMP_COOLDOWN:
            jump_with_duck()
            last_jump_time = current_time

        elif not near_detected and far_detected and (current_time - last_jump_time) > JUMP_COOLDOWN:
            jump_with_duck()
            last_jump_time = current_time

        cv2.rectangle(full_img, (near_x1, near_y1), (near_x2, near_y2), (255, 0, 0), 2)
        cv2.rectangle(full_img, (far_x1, far_y1), (far_x2, far_y2), (0, 255, 0), 2)

        cv2.imshow('Game', full_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        score += 1
        time.sleep(0.01)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
