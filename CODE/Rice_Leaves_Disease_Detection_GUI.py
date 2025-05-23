from PIL import ImageTk
import PIL.Image
from tkinter import *
import tkinter as tk
from tkinter import filedialog
from skimage.io import imread
from skimage.transform import resize
import skimage, pickle
import os

windo = Tk()
windo.configure(background='light yellow')
windo.title("Rice Leaves Disease Detection App")
windo.geometry('1120x820')

# Try to set icon, but continue if it fails
try:
    # Use a relative path or ensure the absolute path is correct
    icon_path = os.path.join('images', 'rice.ico')  # Relative path option
    windo.iconbitmap(icon_path)
except:
    pass  # Continue without icon if it fails

windo.resizable(0, 0)

# Size for displaying Image
w = 650
h = 270
size = (w, h)

def upload_im():
    try:
        global im, resized, cp, path, display, imageFrame, dn1
        imageFrame = tk.Frame(windo)
        imageFrame.place(x=415, y=160)
        path = filedialog.askopenfilename()
        im = PIL.Image.open(path)
        resized = im.resize(size, PIL.Image.LANCZOS)  # Updated from ANTIALIAS
        tkimage = ImageTk.PhotoImage(resized)
        display = tk.Label(imageFrame)
        display.imgtk = tkimage
        display.configure(image=tkimage)
        display.grid()
        dn1 = tk.Label(windo, text='Original🚀 Image', width=20, height=1, fg="white", bg="brown4",
                       font=('times', 22, ' bold '))
        dn1.place(x=570, y=120)
        cp = tk.Button(windo, text='Predict🚀 Disease', bg="brown4", fg="white", width=20,
                      height=1, font=('times', 22, 'italic bold '), command=prediction, activebackground='yellow')
        cp.place(x=570, y=440)
    except Exception as e:
        print(e)
        noti = tk.Label(windo, text='Please upload an Image🚀 File', width=33, height=2, fg="white", bg="brown4",
                       font=('times', 23, ' bold '))
        noti.place(x=454, y=540)
        windo.after(5000, destroy_widget, noti)

def destroy_widget(widget):
    widget.destroy()

def prediction():
    windo.after(2000, destroy_widget, cp)
    def load_image(im_file):
        dimension = (104, 104)
        flat_data = []
        img = skimage.io.imread(im_file)
        img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
        flat_data.append(img_resized.flatten())
        return flat_data
    img = load_image(path)
    try:
        model_path = os.path.join('model', 'Rice_Leaves_Disease_Detection.pkl')  # Relative path option
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
        pred = clf.predict(img)
    except Exception as e:
        print(e)
        noti = tk.Label(windo, text='Model not Found', width=33, height=2, fg="white", bg="brown4",
                       font=('times', 23, ' bold '))
        noti.place(x=454, y=580)
        windo.after(5000, destroy_widget, noti)
        return
    
    labels = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']
    s = [str(i) for i in pred]
    a = int("".join(s))
    lab = str("Predicted Disease is " + labels[a])
    pred = tk.Label(windo, text=lab, width=33, height=2, fg="white",
                   bg="blue",
                   font=('times', 23, ' bold '))
    pred.place(x=454, y=540)
    windo.after(7000, destroy_widget, pred)
    windo.after(7000, destroy_widget, display)
    windo.after(7000, destroy_widget, imageFrame)
    windo.after(7000, destroy_widget, dn1)

# Try to load image, but continue if it fails
try:
    ri = PIL.Image.open(os.path.join('images', 'rice.png'))  # Relative path option
    ri = ri.resize((351, 263), PIL.Image.LANCZOS)  # Updated from ANTIALIAS
    sad_img = ImageTk.PhotoImage(ri)
    panel4 = Label(windo, image=sad_img, bg='white')
    panel4.pack()
    panel4.place(x=20, y=170)
except Exception as e:
    print(f"Could not load rice image: {e}")

up = tk.Button(windo, text='Upload🚀 Image', bg="brown4", fg="white", width=20,
               height=1, font=('times', 22, 'italic bold '), command=upload_im, activebackground='yellow')
up.place(x=20, y=440)

pred = tk.Label(windo, text="Rice Leaves Disease Detection", width=30, height=2, fg="white", bg="dark green",
               font=('times', 25, ' bold '))
pred.place(x=254, y=20)

windo.mainloop()