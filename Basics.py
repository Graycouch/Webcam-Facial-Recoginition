import cv2
import face_recognition

imageElon = face_recognition.load_image_file('ImagesBasic/Elon Musk.jpg')
imageElon = cv2.cvtColor(imageElon, cv2.COLOR_BGR2RGB)
imageElonTest = face_recognition.load_image_file('ImagesBasic/Elon Musk Test.jpg')
imageElonTest = cv2.cvtColor(imageElonTest, cv2.COLOR_BGR2RGB)

faceLocation = face_recognition.face_locations(imageElon)[0]
encodeElon = face_recognition.face_encodings(imageElon)[0]
cv2.rectangle(imageElon, (faceLocation[3], faceLocation[0]), (faceLocation[1], faceLocation[2]), (255, 144, 30), 2)

faceLocationTest = face_recognition.face_locations(imageElonTest)[0]
encodeElonTest = face_recognition.face_encodings(imageElonTest)[0]
cv2.rectangle(imageElonTest, (faceLocationTest[3], faceLocationTest[0]), (faceLocationTest[1], faceLocationTest[2]),
              (255, 144, 30), 2)

results = face_recognition.compare_faces([encodeElon], encodeElonTest)
faceDistance = face_recognition.face_distance([encodeElon], encodeElonTest)
print(results, faceDistance)
cv2.putText(imageElonTest, f'{results} {round(faceDistance[0], 2)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255),
            2)

cv2.imshow('Elon Musk', imageElon)
cv2.imshow('Elon Musk Test', imageElonTest)
cv2.waitKey(0)
