import cv2 as cv
import numpy as np
import zmq

address = "84.237.21.36"
port = 6002

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.setsockopt(zmq.SUBSCRIBE, b"") 
socket.connect(f"tcp://{address}:{port}")

# квадрат 
def square(contour):
    x, y, w, h = cv.boundingRect(contour)
    aspect_ratio = w / h
    if 0.8 < aspect_ratio < 1.25 and w > 30 and h > 30:
        peri = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.04 * peri, True)
        return len(approx) == 4 
    return False

# круг
def circle(contour):
    (_, _), radius = cv.minEnclosingCircle(contour)
    if radius == 0:
        return False
    area_estimated = np.pi * radius * radius
    area_real = cv.contourArea(contour)
    return (area_real / area_estimated) > 0.7

# левое меню 
def draw_stats(img, frame_num, squares, circles, total):
    cv.rectangle(img, (5, 5), (220, 140), (255, 0, 255), -1)
    cv.putText(img, f"Frame: {frame_num}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv.putText(img, f"Squares: {squares}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv.putText(img, f"Circles: {circles}", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv.putText(img, f"Total: {total}", (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

def main():

    cv.namedWindow("client", cv.WINDOW_GUI_NORMAL)

    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    # цвета
    color_ranges = {
        "red": [
            (np.array([0, 70, 50]), np.array([10, 255, 255])),
            (np.array([170, 70, 50]), np.array([180, 255, 255])),
            (np.array([0, 30, 10]), np.array([10, 70, 70])),
            (np.array([170, 30, 10]), np.array([180, 70, 70])),
        ],
        "blue": [
            (np.array([90, 40, 40]), np.array([130, 255, 255])),
            (np.array([90, 30, 10]), np.array([130, 70, 70])),
        ],
        "green": [
            (np.array([40, 30, 30]), np.array([80, 255, 255])),
            (np.array([40, 20, 10]), np.array([80, 70, 70])),
        ],
        "yellow": [
            (np.array([20, 40, 40]), np.array([35, 255, 255])),
            (np.array([20, 20, 10]), np.array([35, 70, 70])),
        ],
        "orange": [
            (np.array([10, 40, 40]), np.array([20, 255, 255])),
            (np.array([10, 20, 10]), np.array([20, 70, 70])),
        ],
        "violet": [
            (np.array([130, 50, 50]), np.array([160, 255, 255])),
            (np.array([130, 20, 10]), np.array([160, 70, 70])),
        ],
        "pink": [
            (np.array([160, 50, 50]), np.array([170, 255, 255])),
            (np.array([160, 20, 10]), np.array([170, 70, 70])),
        ],
        "brown": [
            (np.array([10, 50, 20]), np.array([20, 255, 200])),
        ],
    }

    frame_num = 0
    while True:
        raw = socket.recv()  # кадр
        frame = cv.imdecode(np.frombuffer(raw, np.uint8), cv.IMREAD_COLOR)
        frame_num += 1

        # перевод в HSV
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        v_channel = hsv[:, :, 2]
        hsv[:, :, 2] = clahe.apply(v_channel)

        # объединение масок
        combined_mask = None
        for shades in color_ranges.values():
            for lower, upper in shades:
                mask = cv.inRange(hsv, lower, upper)
                if combined_mask is None:
                    combined_mask = mask
                else:
                    combined_mask = cv.bitwise_or(combined_mask, mask)

        # морфология
        kernel = np.ones((7, 7), np.uint8)
        mask = cv.erode(combined_mask, kernel, iterations=3)
        mask = cv.dilate(mask, kernel, iterations=2)

        # контуры
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        squares = 0
        circles = 0
        total = 0

        for cnt in contours:
            if cv.contourArea(cnt) < 280:
                continue  

            if square(cnt):
                color = (0, 255, 0)  
                label = "Square"
                squares += 1
            elif circle(cnt):
                color = (0, 0, 255) 
                label = "Circle"
                circles += 1
            else:
                color = (255, 255, 0) 
                label = "nothing"

            total += 1
            cv.drawContours(frame, [cnt], -1, color, 2)

            x, y, w, h = cv.boundingRect(cnt)
            cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        draw_stats(frame, frame_num, squares, circles, total)

        cv.imshow("client", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        
    cv.destroyAllWindows()
    socket.close()
    context.term()

if __name__ == "__main__":
    main()
