import cv2

print(cv2.__file__)

image = cv2.imread("happy.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

faces = face_detector.detectMultiScale(gray, 1.1, 5)

for (x, y, w, h) in faces:
    face = image[y:y+h, x:x+w]
    print("Original:", face.shape)
    face = cv2.resize(face, (64, 64))
    face = face / 255.0
    print("Resized:", face.shape)
    print("Min:", face.min())
    print("Max:", face.max())
    cv2.imshow("Face", face)

cv2.waitKey(0)
cv2.destroyAllWindows()