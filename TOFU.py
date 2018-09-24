import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt
import datetime
from keras.preprocessing import image
from keras.models import model_from_json
from keras.utils import to_categorical

def judgement():
    #capture video
    cap = cv2.VideoCapture(0)
    #face_cascade
    HAAR_FILE = '/usr/local/var/pyenv/versions/anaconda3-5.2.0/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml'
    cascade = cv2.CascadeClassifier(HAAR_FILE)
    model = model_from_json(open('model.json', 'r').read())
    model.load_weights('weight.h5')
    while True:
        ret,frame = cap.read()
        cv2.imshow("frame",frame)
        if not ret:
            print('error')
            continue
        if cv2.waitKey(1) & 0xFF == ord('p'):
            cv2.imwrite('frame.jpg',frame)
            image = cv2.imread('frame.jpg')
            image_gray = cv2.imread('frame.jpg', 0)
            face = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=3, minSize=(240, 240))
            if len(face) > 0:
                for x,y,w,h in face:
                    face_cut = image[y:y+h, x:x+w]
                if not os.path.exists('image'):
                    os.mkdir('image')
                now = datetime.datetime.now()
                d = datetime.datetime(now.year,now.month,now.day,now.hour,now.minute,now.second)
                face = face_cut
                cv2.imwrite(f'image/{d}.jpg',face_cut) 
                img = cv2.resize(face_cut, (150, 150))
                img = img/255
                img = np.expand_dims(img, axis=0)
                img_pred = model.predict(img)
                plt.imshow(face)
                plt.title('{:.2f}'.format(float(img_pred[0][1] * 100))+ '%')
                plt.xticks([]),plt.yticks([])
                plt.show() 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    sys.exit()
    print('---END---')

if __name__ == '__main__':
    judgement()
