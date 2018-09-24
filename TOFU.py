import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import model_from_json
from keras.utils import to_categorical

def judgement():
    #capture video
    cap = cv2.VideoCapture(0)
    #face_cascade
    HAAR_FILE = r"c:\Users\pigii\Anaconda3\envs\tensorflow-gpu\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(HAAR_FILE)
    model = model_from_json(open('model.json', 'r').read())
    model.load_weights('weight.h5')
    while True:
        ret,frame = cap.read()
        edframe = frame
        cv2.imshow("frame",edframe)
        if not ret:
            print('error')
            continue
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('p'):
            img = frame.copy()
            img_gray = cv2.imread(img, 0)
            face = cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=3, minSize=(240, 240))
            if len(face) > 0:
                for x,y,w,h in face:
                    face_cut = image[y:y+h, x:x+w]
            cv2.imwrite('0.jpg',face_cut) 
            img = cv2.resize(face_cut, (150, 150))
            img = image.img_to_array(img)
            img = img/255
            img = np.expand_dims(img, axis=0)
            predicts = model.predict(img)
            plt.imshow(img)
            plt.title('{:.2f}'.format(float(img_pred[0][1] * 100))+ '%')
            plt.xticks([]),plt.yticks([])
        cap.release()
        cv2.destroyAllWindows()
        sys.exit()
        print('---END---')

if __name__ == '__main__':
    judgement()
