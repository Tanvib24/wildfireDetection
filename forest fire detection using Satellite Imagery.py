#!/usr/bin/env python
# coding: utf-8

# In[12]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np


# In[13]:


train_dir = r"C:\Users\Tanvibandebuche\Downloads\green AI\archive.zip\train"
valid_dir = r"C:\Users\Tanvibandebuche\Downloads\green AI\archive.zip\valid"
test_dir = r"C:\Users\Tanvibandebuche\Downloads\green AI\archive.zip\test"

train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(64, 64), batch_size=32, class_mode='binary')
valid_generator = valid_datagen.flow_from_directory(valid_dir, target_size=(64, 64), batch_size=32, class_mode='binary')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(64, 64), batch_size=32, class_mode='binary')


# In[ ]:


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[14]:


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[15]:


history = model.fit(train_generator, validation_data=valid_generator, epochs=10, verbose=1)


# In[ ]:


def predict_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((200, 200))
        img = ImageTk.PhotoImage(img)
        image_label.configure(image=img)
        image_label.image = img
        
        img_for_model = Image.open(file_path).resize((64, 64))
        img_array = np.array(img_for_model) / 255.0 
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)[0][0]
        result ="Wildfire" if prediction > 0.5 else "No wildfire"
        result_label.config(text="prediction: "+ result)
root = tk.Tk()
root.title("Forest Fire Detection")
root.geometry("400*400")
btn = tk.Button(root, text="Upload Image", command=predict_image)
btn.pack(pady=20)
image_label = tk.label(root)
image_label.pack()
result_label = tk.Label(root, text="Prediction: ",font=("Helvetica",16))
result_label.pack(pady=20)
root.mainloop()


# In[ ]:




