import cv2
import numpy as np
cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE,1)
capture.set(cv2.CAP_PROP_EXPOSURE,-3)

def censore(image,size=(5,5)):
    result=np.zeros_like(image)
    stepy=result.shape[0]//size[0]
    stepx=result.shape[1]//size[1]
    for y in range(0,image.shape[0],stepy):
        for x in range(1,image.shape[1],stepx):
            for c in range(0,image.shape[2]):
                result[y:y+stepy,x:x+stepx,c]=np.mean(image[y:y+stepy,x:x+stepx,c])
    return result

face_cascade = cv2.CascadeClassifier("deal_with_it\haarcascade-frontalface-default.xml")
eye_cascade = cv2.CascadeClassifier("deal_with_it\haarcascade-eye.xml")

while capture.isOpened():
    
    ret, frame = capture.read()
    blurred = cv2.GaussianBlur(frame, (11,11), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    faces = eye_cascade.detectMultiScale(gray, scaleFactor=3, minNeighbors=9)
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        new_w=w#int(w*1.5)
        new_h=h#int(h*1.5)
        x-=w//4
        y-=h//4
        try:
            roi=frame[y:y+new_h,x:x+new_w]
            censored=censore(roi,(10,10))
            frame[y:y+new_h,x:x+new_w]=censored
        except ValueError:
            pass
    key = chr(cv2.waitKey(1) & 0xFF)
    if key == "q":
        break
    cv2.imshow("Camera", frame)
capture.release()
cv2.destroyAllWindows()
