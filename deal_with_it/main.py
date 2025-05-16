import cv2
import numpy as np

cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
capture.set(cv2.CAP_PROP_EXPOSURE, -3)

face_cascade = cv2.CascadeClassifier("deal_with_it/haarcascade-frontalface-default.xml")
eye_cascade = cv2.CascadeClassifier("deal_with_it/haarcascade-eye.xml")

glasses = cv2.imread("deal_with_it/glasses.png") 

# цензура
def censore(image, size=(5, 5)):
    result = np.zeros_like(image)
    stepy = result.shape[0] // size[0]
    stepx = result.shape[1] // size[1]
    for y in range(0, image.shape[0], stepy):
        for x in range(0, image.shape[1], stepx):
            for c in range(0, image.shape[2]):
                result[y:y+stepy, x:x+stepx, c] = np.mean(image[y:y+stepy, x:x+stepx, c])
    return result

# приклеивание очков 
def overlay_glasses(frame, eyes, glasses_img):
    if len(eyes) < 2:
        return frame

    # сорт по x-координате чтобы найти левый и правый глащ
    eyes = sorted(eyes, key=lambda e: e[0])
    (x1, y1, w1, h1), (x2, y2, w2, h2) = eyes[:2]

    # центр между глазами
    center_x = (x1 + x2 + w1//2 + w2//2) // 2
    center_y = (y1 + y2 + h1//2 + h2//2) // 2

    # размер очков
    glasses_width = int(2.2 * abs((x2 + w2//2) - (x1 + w1//2)))
    glasses_height = int(glasses_img.shape[0] * glasses_width / glasses_img.shape[1])

    # изменение размера очков
    resized_glasses = cv2.resize(glasses_img, (glasses_width, glasses_height))

    # маска очков 
    gray = cv2.cvtColor(resized_glasses, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)

    top_left_x = center_x - glasses_width // 2
    top_left_y = center_y - glasses_height // 2

    # границы
    if top_left_x < 0 or top_left_y < 0 or top_left_x + glasses_width > frame.shape[1] or top_left_y + glasses_height > frame.shape[0]:
        return frame

    roi = frame[top_left_y:top_left_y+glasses_height, top_left_x:top_left_x+glasses_width]

    # прикрепляем
    bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    fg = cv2.bitwise_and(resized_glasses, resized_glasses, mask=mask)
    combined = cv2.add(bg, fg)

    frame[top_left_y:top_left_y+glasses_height, top_left_x:top_left_x+glasses_width] = combined
    return frame

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (fx, fy, fw, fh) in faces:
        roi_gray = gray[fy:fy+fh, fx:fx+fw]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=10)
        eyes = [(ex+fx, ey+fy, ew, eh) for (ex, ey, ew, eh) in eyes]

        # приклеиваем очки 
        frame = overlay_glasses(frame, eyes, glasses)

    key = chr(cv2.waitKey(1) & 0xFF)
    if key == "q":
        break
    cv2.imshow("Camera", frame)

capture.release()
cv2.destroyAllWindows()