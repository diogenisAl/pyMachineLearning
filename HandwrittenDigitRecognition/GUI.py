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
    print("WIP")


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
