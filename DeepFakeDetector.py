import os

import cv2
import time
import pyautogui
from main import load_and_predict
# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def capture_and_save_faces():
    while True:
        time.sleep(10)
        # Take screenshot
        screenshot = pyautogui.screenshot()
        screenshot.save('screenshot.png')

        img = cv2.imread('screenshot.png')

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        faces_count = 0
        deepfake_count = 0

        for i, (x, y, w, h) in enumerate(faces):
          if faces_count>5:
              break
          if w >= 100 and h >= 100 :
            faces_count += 1
            padding = 25
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1], w + 2 * padding)
            h = min(img.shape[0], h + 2 * padding)

            face_img = img[y:y + h, x:x + w]

            face_img = cv2.resize(face_img, (256, 256))

            cv2.imwrite(f'face{faces_count}.png', face_img)

            prediction = load_and_predict(f'face{faces_count}.png')

            if prediction=="Deepfake":
                break

            #os.remove(f'face.png')

        # Display the captured frame (optional)
        cv2.imshow('Captured Frame', img)

        # Wait for 5 seconds or until a key is pressed
        # if cv2.waitKey(5000) & 0xFF == ord('q'):
        #     break

        # while faces_count != 0:
        #     os.remove(f'face_{faces_count}.png')
        #     faces_count -= 1

        # Close OpenCV window
        cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_and_save_faces()
