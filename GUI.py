from tkinter import *
from tkinter import filedialog
import os
import tensorflow as tf
import keras
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from PIL.ImageTk import PhotoImage
from netconfig import NetConfig
from densenet import DenseNet
import cv2

root = Tk()
parent_frame = Frame(root, bg="#000", width=100, height=100)
parent_frame.pack(side=RIGHT, pady=0, padx=(0, 20))

lbl_frame = Frame(parent_frame, bg="#000")
lbl_frame.pack(side=TOP, padx=(0, 0), pady=(0, 10))

load_frame = Frame(parent_frame, bg="#120009")

check_frame = Frame(parent_frame, bg="#120009")
check_result = Label(check_frame, text="Our System Opinion", font='Helvetica 15 bold', height=2, width=20,
                     fg="#120009")
path = "defImg.jpg"
img = ImageTk.PhotoImage(Image.open(path))
panel = tk.Label(root, image=img, bg="#000")
panel.pack(side=LEFT, fill="both", expand="yes", pady=20, padx=20)
variable = StringVar(load_frame)
part_options = OptionMenu(load_frame, variable, "HUMERUS", "FOREARM", "HAND", "SHOULDER", "WRIST", "FINGER", "ELBOW")

image_list = []

conf = NetConfig()
dense_model = DenseNet(conf)

# ============================   Model-based Objects   ============================




def show_loading_hint():
    pass


def hide_loading_hint():
    pass


def render():

    # ============================   check frame   ============================
    check_frame.pack(side=RIGHT, padx=(10, 20), pady=(10, 10))
    btn_check = Button(check_frame, text="Check", command=check_callback, height=1, width=15, fg="#120009",
                       font='Helvetica 15 bold')
    btn_check.pack(side=TOP, padx=10, pady=10)


    check_result.pack(side=BOTTOM, padx=10, pady=10)

    # ============================   Label frame  ============================
    lbl_img_path = "lblImg.jpg"
    lbl_img: PhotoImage = ImageTk.PhotoImage(Image.open(lbl_img_path))
    lbl_panel = tk.Label(lbl_frame, image=lbl_img, bg="#120009")
    lbl_panel.pack(fill="both", expand="yes", pady=10, padx=10, ipadx=5, ipady=5)
    lbl_panel.configure(image=lbl_img)
    lbl_panel.image = lbl_img

    # ============================   Loading frame   ============================
    load_frame.pack(side=LEFT, padx=(20, 10), pady=(10, 10))
    btn_load = Button(load_frame, text="Load Image", command=upload_image_callback, height=1, width=15, fg="#120009",
                      font='Helvetica 15 bold')
    btn_load.pack(side=TOP, padx=10, pady=10)

    label1 = Label(load_frame, text="Select Part For Best Result", fg="#120009", font='Helvetica 10 bold')
    label1.pack(padx=10, pady=10)
    variable.set("General")  # default value
    part_options.pack()

    # ============================   Root Tuning   ============================
    root.title("bone fracture")
    root.geometry("1300x700")
    root.configure(background='#120009')
    root.mainloop()


def upload_image_callback():
    selected_img = filedialog.askopenfilename(initialdir=os.getcwd(), title="select image",
                                              filetypes=(("All Files", "*.*"), ("JPG File", "*.jpg"),
                                                         ("PNG File", "*.png")))
    img_selected = Image.open(selected_img)
    img_selected: PhotoImage = ImageTk.PhotoImage(img_selected)
    panel.configure(image=img_selected)
    panel.image = img_selected
    image_list.clear()
    image_list.append(selected_img)


def check_callback():

    part = variable.get()

    # TODO
    show_loading_hint()

    # loading selected model in option list
    try:
        print("loading {} model".format(part))
        dense_model.load_model_by_path(r"./saved_model/{}/{}.h5".format(part, part))
    except:
        print("model not found, loading HUMERUS model instead.!")
        dense_model.load_model_by_path(r"saved_model\HUMERUS\HUMERUS.h5")
    print("model loaded")

    # TODO
    hide_loading_hint()
    img_array = tf.keras.preprocessing.image.load_img(
        image_list[0], target_size=(conf.InputShape, conf.InputShape), color_mode="grayscale"
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img_array)
    # Create a batch
    img_array = tf.expand_dims(img_array, 0)

    '''    img_array = cv2.imread(image_list[0], 0)
    img_array = cv2.resize(img_array, dsize=(conf.InputShape, conf.InputShape))
    img_array = np.reshape(img_array, (conf.InputShape, conf.InputShape, 1))'''

    print("image shape : ", img_array.shape)
    predictions = dense_model.predict(img_array)
    print(predictions)
    if predictions[0][0] > predictions[0][1]:
        set_color(1)
    else:
        set_color(0)
    # 1- get the last selected pic
    #selectedimg = image_list[len(image_list) - 1]
    # 2- test this pic
    # 3- call setcolor due to the output of the test
    pass


def set_color(found):
    if found:
        check_result.configure(bg="red", text="Oops.Have Fracture!")
    else:
        check_result.configure(bg="green", text="Have No fracture")


if __name__ == '__main__':
    render()


    """

    exitBTN = Button(parent_frame, text="Exit", command=lambda: exit())
    exitBTN.pack(side=tk.RIGHT, padx=20)"""


