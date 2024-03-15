import cv2
import os
import numpy as np
from mtcnn import MTCNN
import matplotlib.pyplot as plt

class FACELOADING:
    def __init__(self, directory):
        self.directory = directory
        self.set_target_size(160, 160)
        self.X = []
        self.Y = []
        self.detector = MTCNN()

    def set_target_size(self, width, height):
        self.target_width = width
        self.target_height = height

    def extract_face(self, filename):
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(img)
        if faces:
            x, y, w, h = faces[0]['box']
            x, y = abs(x), abs(y)
            face = img[y:y+h, x:x+w]
            return cv2.resize(face, (self.target_width, self.target_height))  # Resize the face image
        else:
            return None

    def load_faces(self, dir):
        FACES = []
        for im_name in os.listdir(dir):
            try:
                path = os.path.join(dir, im_name)
                single_face = self.extract_face(path)
                if single_face is not None:
                    FACES.append(single_face)
            except Exception as e:
                pass
        return FACES

    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = os.path.join(self.directory, sub_dir)
            FACES = self.load_faces(path)
            labels = [sub_dir for _ in range(len(FACES))]
            print(f"loaded successfully: {len(labels)}")
            self.X.extend(FACES)
            self.Y.extend(labels)
        return np.asarray(self.X), np.asarray(self.Y)

    def plot_images(self):
        plt.figure(figsize=(18,16))
        for num, image in enumerate(self.X):
            ncols = 3
            nrows = len(self.Y) // ncols
            plt.subplot(nrows, ncols, num+1)
            plt.imshow(image)
            plt.axis('off')


faceloading = FACELOADING("C:\\Users\\Joseph Kariuki\\dataset")
X, Y = faceloading.load_classes()

plt.figure(figsize=(16,12))
for num,image in enumerate(X):
    ncols=3
    nrows=len(Y)//ncols+1
    plt.subplot(nrows,ncols,num+1)
    plt.imshow(image)
    plt.axis('off')

from keras_facenet import FaceNet
embedder=FaceNet()

def get_embedding(face_img):
    face_img=face_img.astype('float32') #3D(160x160x3)
    face_img=np.expand_dims(face_img,axis=0)
    yhat=embedder.embeddings(face_img)
    return yhat[0]
EMBEDDED_X=[]
for img in X:
    EMBEDDED_X.append(get_embedding(img))
EMBEDDED_X=np.asarray(EMBEDDED_X)    

np.savez_compressed('faces_embeddings_done_4classes.npz',EMBEDDED_X, Y)
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
encoder.fit(Y)
Y=encoder.transform(Y)
#SVM MODEL
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(EMBEDDED_X,Y, shuffle=True, random_state=17 )

from sklearn.svm import SVC
model=SVC(kernel='linear', probability=True)
model.fit(X_train, Y_train)
ypreds_train=model.predict(X_train)
ypreds_test=model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(Y_train,ypreds_train)

#using test image
t_im=cv2.imread('d.jpg')
t_im=cv2.cvtColor(t_im, cv2.COLOR_BGR2RGB)
detector.detect_faces(t_im)[0]['box']


t_im = t_im[y:y+h, x:x+w]
if not t_im.size:
    print("Error: Cropped image is empty")
else:
    t_im = cv2.resize(t_im, (160, 160))
    test_im = get_embedding(t_im)


test_im= [test_im]
ypreds=model.predict(test_im)

encoder.inverse_transform(ypreds)

