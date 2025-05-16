import cv2
import numpy as np

cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
capture.set(cv2.CAP_PROP_EXPOSURE, -3)

face_cascade = cv2.CascadeClassifier("deal_with_it/haarcascade-frontalface-default.xml")
eye_cascade = cv2.CascadeClassifier("deal_with_it/haarcascade-eye.xml")

glasses = cv2.imread("deal_with_it/glasses.png", cv2.IMREAD_UNCHANGED)

def censore(image, eyes, glasses_img):
    # проверка на 2 глаза и альфаканал
    if len(eyes) < 2 or glasses_img.shape[2] != 4:
        return image

    # сорт по горизонатали 
    eyes = sorted(eyes, key=lambda e: e[0])
    (x1, y1, w1, h1), (x2, y2, w2, h2) = eyes[:2]

    # центр между глазами 
    center_x = (x1 + x2 + w1 // 2 + w2 // 2) // 2
    center_y = (y1 + y2 + h1 // 2 + h2 // 2) // 2

    # ширина очков
    glasses_width = int(2.2 * abs((x2 + w2//2) - (x1 + w1//2)))
    glasses_height = int(glasses_img.shape[0] * glasses_width / glasses_img.shape[1])

    # верх левый угол размещаем очки 
    top_left_x = center_x - glasses_width // 2
    top_left_y = center_y - glasses_height // 2

    # выход за границы 
    if (top_left_x < 0 or top_left_y < 0 or
        top_left_x + glasses_width > image.shape[1] or
        top_left_y + glasses_height > image.shape[0]):
        return image

    # изменение размера очков 
    resized_glasses = cv2.resize(glasses_img, (glasses_width, glasses_height))

    # маска програчности для очков 
    alpha = resized_glasses[:, :, 3] / 255.0

    # прикрепление очков
    for c in range(3):
        image[top_left_y:top_left_y+glasses_height, top_left_x:top_left_x+glasses_width, c] = (
            alpha * resized_glasses[:, :, c] +
            (1 - alpha) * image[top_left_y:top_left_y+glasses_height, top_left_x:top_left_x+glasses_width, c]
        ).astype(np.uint8)

    return image

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (fx, fy, fw, fh) in faces:
        # область лица
        roi_gray = gray[fy:fy+fh, fx:fx+fw]
        roi_color = frame[fy:fy+fh, fx:fx+fw]

        # находим глаза
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=10)

        eyes = [(ex + fx, ey + fy, ew, eh) for (ex, ey, ew, eh) in eyes]

        # прикрепляем  очки 
        frame = censore(frame, eyes, glasses)

    if chr(cv2.waitKey(1) & 0xFF) == "q":
        break
    cv2.imshow("Camera", frame)

capture.release()
cv2.destroyAllWindows()