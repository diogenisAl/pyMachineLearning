# Modules for the AI
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras import backend as K

# Get the data from keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# # Plot some of the data
# import matplotlib.pyplot as plt
# for i in range(1, 7):
#     plt.subplot(int('23' + str(i)))
#     plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))


# the data, shuffled and split between train and test sets
img_rows = 28
img_cols = 28

# Images in the dataset are represented as a 28×28 matrix containing grayscale pixel values
# reshape format: [samples][width][height][channels]
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)

# Convert pixel values to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize inputs
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Create the model
num_classes = 10
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train,
          batch_size=200,
          epochs=3,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print(score)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('mnisk.h5')

# Modules for the GUI
import os
import PIL
import cv2
import glob
import numpy as np
from tkinter import *
from PIL import Image, ImageDraw, ImageGrab

def clear_widget():
    global cv
    cv.delete("all")

# Every time the left mouse button is vlivked the activate event method gets called
def activate_event(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y

# Every time the left mouse button is clicked and moves the draw_lines event methof gets called
def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    # Draw the line
    cv.create_line((lastx, lasty, x, y), width=8, fill='black',
                  capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = event.x, event.y

def Predict_Digit():
    global image_number
    predictions = []
    percentages = []

    filename = f'image_{image_number}.png'
    widget=cv

    # get widget coordinates
    x = root.winfo_rootx() + widget.winfo_x()
    y = root.winfo_rooty() + widget.winfo_y()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()

    ImageGrab.grab().crop((x,y,x1,y1)).save(filename)

    #
    image = cv2.imread(filename, cv2.IMREAD_COLOR)

    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Make the image ONLY black and white using Otsu thresholding
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # fincContour helps in extracting the contours from the image
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    for cnt in contours:
        # Get bounding box and extract ROT
        x, y, w, h = cv2.boundingRect(cnt)

        # Create rectangle
        cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 1)
        top = int(0.05 * th.shape[0])
        bottom = top
        left = int(0.05 * th.shape[1])
        right = left
        th_up = cv2.copyMakeBorder(th, top, bottom, left, right, cv2.BORDER_REPLICATE)

        # Extract the image ROI
        roi = th[y-top:y+h+bottom, x-left:x+w+right]

        # Resize roi image to 28x28
        try:
            img = cv2.resize(roi, (28,28), interpolation=cv2.INTER_AREA)
        except:
            continue
        # Reshape the image to support model's input
        img = img.reshape(1, 28, 28, 1)

        # Normalize the image
        img = img/255.0

        # Predict the result
        pred = model.predict([img])[0]

        # Get the indices of the maximum values
        final_pred = np.argmax(pred)
        data = str(final_pred) + " " + str(int(max(pred) * 100)) + "%"

        # Add the label on top of the box
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontscale = 0.5
        color = (255, 0, 0)
        thickness = 1
        cv2.putText(image, data, (x,y-5), font, fontscale, color, thickness)

    # Show the results on a new window
    cv2.imshow("image", image)
    cv2.waitKey


# Create the main window
root = Tk()
root.resizable(0, 0)
root.title("Handwritten Digit Recognition App")

lastx, lasty = None, None
image_number = 0

# Create the canvas for drawing
cv = Canvas(root, width=640, height=480, bg='white')
cv.grid(row=0, column=0, pady=2, sticky=W, columnspan=2)

# Similar to pygame you can handle events using Tkinter
cv.bind('<Button-1>', activate_event)

# Add Buttons and labels
btn_save = Button(text="Predict Digit", command=Predict_Digit)
btn_save.grid(row=2, column=1, pady=1, padx=1)

btn_clear = Button(text="Clear", command=clear_widget)
btn_clear.grid(row=2, column=0, pady=1, padx=1)

# run the main loop, similar to pygame
root.mainloop()