import cv2
import time

face = []
eye = []

faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')

img = cv2.imread('/Users/jganeeva/Desktop/detect_dataset/12.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
start = time.time()
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.9,
    minNeighbors=4,
    minSize=(30, 30)
)
end = time.time()
print("Execution Time (in seconds) :")
print("Haar: ", format(end - start, '.2f'))

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
    roi_gray = gray[y:y + h + 1, x:x + w + 1]
    roi_color = img[y:y + h + 1, x:x + w + 1]
    eyes = eyeCascade.detectMultiScale(
        roi_gray,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(5, 5),
    )
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 1)
        cv2.rectangle(roi_gray, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 1)
        eye_tmp = roi_gray[ey:ey + eh + 1, ex:ex+ew+1]
        eye.append(eye_tmp)
    face.append(roi_gray)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()