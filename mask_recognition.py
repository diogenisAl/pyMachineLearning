import pandas as pd
import numpy as np
import cv2
import json
import os
import matplotlib.pyplot as plt
import random
import seaborn as sns
from keras.models import Sequential
from keras.models import load_model
from keras import optimizers
from keras import backend as K
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

directory = "data/Medical mask/Medical mask/Medical Mask/annotations"
image_directory = "data/Medical mask/Medical mask/Medical Mask/images"
df = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/submission.csv")

# Create some auxilliary functions
def getJSON(filePathandName):
    with open(filePathandName,'r') as f:
        return json.load(f)

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])
    return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))

jsonfiles= []
for i in os.listdir(directory):
    jsonfiles.append(getJSON(os.path.join(directory,i)))

df = pd.read_csv("data/train.csv")
img_size = 124
mask = ['face_with_mask']
non_mask = ["face_no_mask"]
labels = {'mask':0,'without mask':1}

# Create a data list with all the images and their labels
data = []
for i in df["name"].unique():
    f = i+".json"
    for j in getJSON(os.path.join(directory,f)).get("Annotations"):

        if j["classname"] in mask:  # If the picture contains a mask
            x,y,w,h = j["BoundingBox"]  # Get the coords of where the face with the mask is
            img = cv2.imread(os.path.join(image_directory,i),1) # Open the image that coresponds to the json file
            img = img[y:h,x:w] # Crop the face and delete everything else
            img = cv2.resize(img,(img_size,img_size)) # Resize to save space
            data.append([img,labels["mask"]]) # Add the image and a 1 to show that it contains a mask

        if j["classname"] in non_mask: # If the picture does not contain a mask
            x,y,w,h = j["BoundingBox"] # Get the coords of where the face is
            img = cv2.imread(os.path.join(image_directory,i),1) # Open the image that coresponds to the json file
            img = img[y:h,x:w]  # Crop the face and delete everything else
            img = cv2.resize(img,(img_size,img_size))# Resize to save space
            data.append([img,labels["without mask"]]) # Add the image and a 0 to show that it does not contain a mask

random.shuffle(data)

# Split the data list in X and Y lists, holding the images and labels respectively
X = []
Y = []

# Split the features and their labels to two different lists
for features,label in data:
    X.append(features)
    Y.append(label)

X = np.array(X)/255.0
X = X.reshape(-1,124,124,3)
Y = np.array(Y)


# Create the model
model = Sequential() # Choose a model. Sequential models are best for simple classifications

# Add the layers
model.add(Conv2D(32, (3, 3), padding = "same", activation='relu', input_shape=(124,124,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# model.compile is used to save metrics for the model's perfomance
model.compile(loss='binary_crossentropy', optimizer='adam' ,metrics=['accuracy'])

# Split the dataset with the labels in training and testing sets
xtrain,xval,ytrain,yval=train_test_split(X, Y, train_size=0.8,random_state=0)

# Create a dataset from the images
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
)

datagen.fit(xtrain)
# Train the model (This will take a while)
history = model.fit_generator(datagen.flow(xtrain, ytrain, batch_size=32),
                    steps_per_epoch=xtrain.shape[0]//32,
                    epochs=50,
                    verbose=1,
                    validation_data=(xval, yval))
model.save('model_for_masks')


# Test the model against random images
test_images = ['1114.png', '1504.jpg', '0072.jpg', '0012.jpg', '0353.jpg', '1374.jpg']
cvNet = cv2.dnn.readNetFromCaffe('data/deploy.prototext', 'data/weights.caffemodel')
reconstructed_model = load_model("model")

# Similar to before we created the model, we need to manipulate the images so that the model can accept them
gamma = 2.0
fig = plt.figure(figsize = (14,14))
rows = 3
cols = 2
axes = []
assign = {'0':'Mask','1':"No Mask"}
for j,im in enumerate(test_images):
    image =  cv2.imread(os.path.join(image_directory,im),1)
    image =  adjust_gamma(image, gamma=gamma)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    cvNet.setInput(blob)
    detections = cvNet.forward()
    for i in range(0, detections.shape[2]):
        try:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            frame = image[startY:endY, startX:endX]
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                im = cv2.resize(frame,(img_size,img_size))
                im = np.array(im)/255.0
                im = im.reshape(1,124,124,3)
                result = reconstructed_model.predict(im)
                if result>0.5:
                    label_Y = 1
                else:
                    label_Y = 0
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(image, assign[str(label_Y)], (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (36,255,12), 2)

        except:
            pass
    axes.append(fig.add_subplot(rows, cols, j+1))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
