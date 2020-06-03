from mtcnn import MTCNN
import cv2
import time

image = cv2.imread("/Users/jganeeva/Desktop/detect_dataset/12.jpg")
eyeCascade = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')

detector = MTCNN()

start = time.time()
face = detector.detect_faces(image)
end = time.time()
print("Execution Time (in seconds) :")
print("MTCNN : ", format(end - start, '.2f'))

for i in range(len(face)):
    x1, y1, w, h = face[i]['box']
    x2, y2 = x1 + w, y1 + h
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
    roi_color = image[y1:y2 + 1, x1: x2 + 1]

    eyes = eyeCascade.detectMultiScale(
        roi_color,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(5, 5),
    )
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 1)
        eye_tmp = roi_color[ey:ey + eh + 1, ex:ex + ew + 1]


cv2.imshow('img', image)
cv2.waitKey(0)