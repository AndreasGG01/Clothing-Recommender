# Import Required Libraries
from tkinter import Tk, Label, Toplevel, messagebox, filedialog, Entry, Frame, Button
from PIL import Image, ImageTk
import cv2
import os
import glob
import shutil
import os
import pandas as pd
import tensorflow as tf
import zipfile
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import losses
import numpy as np
from keras.applications.vgg16 import VGG16
# import cv2 <-- To use openCV function/methods

conv_base = VGG16(weights='imagenet',include_top=False, input_shape=(300, 300, 3))
model = models.Sequential()
model.add(conv_base)

model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(8, activation='sigmoid'))

model.load_weights('C:\\Users\\Andre\\checkpoints_less_images\\weights.50-0.75.hdf5')

# Create a Window.
MyWindow = Tk() # Create a window
MyWindow.title("Clothing Recommender") # Change the Title of the GUI
MyWindow.geometry('600x300') # Set the size of the Windows

# Create the GUI Component but dont display or add them to the window yet.
MyTitle = Label(MyWindow, text = "Clothing Recommender", font=("Helvetica", 20))
InfoLabel = Label(MyWindow, text = "Steps to use the Clothing Recommender:", font=("Arial Bold", 15)) # Addind information on the Text Entry box.
Step1Label = Label(MyWindow, text = '1. Take a photo of your clothing, top or bottom, with the "Snapshot" button ', font=("Arial Bold", 10)) # Addind information on the Text Entry box.
Step2Label = Label(MyWindow, text = '2. Click the "Classify Clothing" button for the Clothing Recommender AI to identify your clothing and make a store recommendation', font=("Arial Bold", 10)) # Addind information on the Text Entry box.
userEntry = Entry(MyWindow, width = 20) # Allows to Enter single line of text
start_frame= Frame(MyWindow, padx=5, pady=5)

# Create the Custom Methods for Processing the Images/Video using DL model
# Define function to show frame
def show_frames():
   # Get the latest frame and convert into Image
   cv2image= cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2RGB)
   global image_save 
   image_save = cv2image
   global img_array
   img_array = Image.fromarray(cv2image)
   # Convert image to PhotoImage
   imgtk = ImageTk.PhotoImage(image = img_array)
   frame_video.imgtk = imgtk
   frame_video.configure(image=imgtk)
   # Repeat after an interval to capture continiously
   frame_video.after(20, show_frames)

# Open Image Function using OpenCV
def openImg(filename):
    messagebox.showinfo("Image to Show", filename)
    # Open the image using OPENCV
    # img = imread(filename)
    # cv.imshow(img)



# Create Event Methods attached to the button etc.
def BttnOpen_Clicked():
    messagebox.showinfo("Info", "Save The Image")
    # Use the File Dialog component to Open the Dialog box to select files
    file = filedialog.askdirectory()
    os.chdir(file)
    cv2.imwrite("capturedFrame.jpg", image_save)
    messagebox.showinfo("File Selected", file)

def BttnProcess_Clicked():
    messagebox.showinfo("Info", "Classification")
    # Read and process images/frame using your DL model here <--
    # Testing
    #messagebox.showwarning("Invalid Input","Image is having an invalid format") # Showing Warning not very Critcal 
    #messagebox.showerror("Invalid Input","Image is having an invalid format") # Showing Error, very Critcal 
    #classifcationResult = "CAT"
    #messagebox.showinfo("Classfication Result", classifcationResult)
    img_predict = cv2.imread('C:\\Users\\Andre\\img_test\\capturedFrame.jpg')
    img_predict = cv2.resize(img_predict,(300,300))
    img_predict = np.reshape(img_predict,[1,300,300,3])
    pred = model.predict(img_predict)
    result_pred = pred.argmax(axis=-1)
    result_recommendation = "" # model.predict(file) for example
    result_classification = ""
    if(result_pred == 0):
        result_classification = "Dress"
        result_recommendation = "Basque - Cord Tiered Dress"
    elif(result_pred == 1):
        result_classification = "Hoodie"
        result_recommendation = "Champion - Script Black Hoodie"
    elif(result_pred == 2):
        result_classification = "Jacket"
        result_recommendation = "North Face - Puffer Jacket"
    elif(result_pred == 3):
        result_classification = "Jeans"
        result_recommendation = "Levis - Straight Jeans"
    elif(result_pred == 4):
        result_classification = "Shorts"
        result_recommendation = "Ralph Lauren - Draw String Shorts"
    elif(result_pred == 5):
        result_classification = "Skirt"
        result_recommendation = "Urban Outfitters - Maisie Midi Skirt"
    elif(result_pred == 6):
        result_classification = "Tank"
        result_recommendation = "Gymshark - Tank Top"
    elif(result_pred == 7):
        result_classification = "Tee"
        result_recommendation = "Tommy Hilfiger - Logo Short Sleeve"
    Classification_text = "Identified Clothing: " + result_classification 
    recommendation_text = "Clothing Recommendation: " + result_recommendation 
       # Concatenate the result class to the Label on the Window
    ClassificationLabel.configure(text = Classification_text)
    recommendation_Label.configure(text = recommendation_text)

def Bttnclose_Clicked():
    MyWindow.destroy()

def open_new_window():
    global frame_video, ClassificationLabel, recommendation_frame, recommendation_Label, cap

    new_window = Toplevel()
    new_window.title("Clothing Recommender") # Change the Title of the GUI
    new_window.geometry('900x600') # Set the size of the Windows

    frame_title = Label(new_window, text = "View Window", pady= 10, font=("Helvetica", 20))
    image_frame= Frame(new_window, padx=5, pady=5)
    recommendation_frame= Frame(new_window, padx=5, pady = 5)
    frame_label = Label(image_frame, text = "No Video Signal", bg="Light grey")
    frame_video = Label(image_frame)
    ClassificationLabel = Label(recommendation_frame, text = "Identified Clothing: ", justify = "left", pady=30, font=("Arial Bold", 10))
    recommendation_Label = Label(recommendation_frame, text = "Clothing Recommendation: ", justify = "left", pady=30,font=("Arial Bold", 10))
    cap = cv2.VideoCapture(0)

    # Add the Components create previsously to the window
    frame_title.grid(column=0, row=0)
    #frame_label.place(x=200, y=375, anchor="center")
    image_frame.grid(column=0, row=1) # Adding the Label
    frame_video.pack()
    openBttn = Button(new_window, text="Classify Clothing", command=BttnProcess_Clicked)
    openBttn.grid(column=0, row=2) # Adding the Open Button
    recordBttn = Button(new_window, text="Snapshot", width = 30, command = BttnOpen_Clicked)
    recordBttn.grid(column=1, row=0)
    closeBttn = Button(new_window, text="close", width = 30, command = Bttnclose_Clicked)
    closeBttn.grid(column=1, row=2)
    # Adding the label to display classfication result
    recommendation_frame.grid(column=1, row=1)
    ClassificationLabel.pack()
    recommendation_Label.pack()
    show_frames()
    MyWindow.wm_state('iconic')

MyTitle.place(relx=.5, rely=.1, anchor = 'center')
InfoLabel.place(relx=.5, rely=.5,anchor= 'center')
Step1Label.place(relx=.5, rely=.6,anchor= 'center')
Step2Label.place(relx=.5, rely=.7,anchor= 'center')
startBttn = Button(MyWindow, text="Start", width = 30, command=open_new_window)
startBttn.place(relx=.5, rely=.9, anchor= 's')

# Calling the maninloop()
MyWindow.mainloop()
