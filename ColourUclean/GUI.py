from tkinter import *
from tkinter.colorchooser import  *
from tkinter import filedialog
from PIL import Image, ImageTk

from skimage.color import rgb2lab, lab2rgb
import numpy as np
import sys

from matplotlib import pyplot as plt
from model import *

import torch

class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master

        self.loaded_image = None
        self.image_data = None
        self.LAB_image_data = None

        self.image_label = Label(self)
        self.image_label.bind("<Button-1>", self.draw_px)
        self.image_label.place(x=0, y=0)

        self.selected_colour = (255, 0, 0)
        self.color_hex = "#FF0000"

        self.transformations = {"scale" : 1.0}

        self.model = None

    #----------------------
    # Initialize window
    #----------------------
    def init_window(self):
        # Name the Window
        self.master.title("Colour U")
        self.pack(fill=BOTH, expand=1)

        # Create the Menu ------------------------------------------------------------------------------------
        menubar = Menu(self.master)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open", command=self.load_img)
        filemenu.add_command(label="Save", command=self.save_file)
        menubar.add_cascade(label="File", menu=filemenu)
        self.master.config(menu=menubar)

        # Buttons ---------------------------------------------------------------------------------------------
        # Frame for the buttons
        b_frame = Frame()
        b_frame.pack(side = LEFT)

        # Actual buttons
        callGAN = Button(text="ColourU", width=10, height=1, command=self.callmodel)
        callGAN.pack(in_=b_frame, side=LEFT, padx=20)
        choosecaller = Button(text="Choose Colour", width=18, height=1, command=self.choose_colour)
        choosecaller.pack(in_ = b_frame, side=LEFT)
        zoomout = Button(text="[-]", width=5, height = 1, command=self.zoom_out)
        zoomout.pack(in_ = b_frame, side=LEFT)
        zoomin = Button(text="[+]", width = 5, height = 1, command=self.zoom_in)
        zoomin.pack(in_ = b_frame, side=LEFT)
        self.currentColour = Label(text='Colour', width=10, height = 1, bg=self.color_hex)
        self.currentColour.pack(in_=b_frame, side=LEFT, padx=40)

        # Import Model
        self.model = torch.load("salamander_fish_G.pt", map_location=torch.device('cpu'))

    # ----------------------
    # Load Image
    # ----------------------
    def load_img(self):
        path = filedialog.askopenfilename(initialdir = "/", title= "Select file", filetypes=(("jpeg files", "*.JPEG"), ("all files", "*.*")))

        self.image_data = rgb2lab(plt.imread(path))
        self.image_data[:, :, 1:] = 0
        self.LAB_image_data = np.copy(self.image_data)
        self.image_data = (255*lab2rgb(self.image_data)).astype(int)
        self.loaded_image = Image.fromarray(self.image_data.astype('B'))

        self.render_image()

    # ----------------------
    # Save File
    # ----------------------
    def save_file(self):
        path = filedialog.asksaveasfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.JPEG"),("all files","*.*")))

        #self.transformed_image.save(path)

    # ----------------------
    # Choose Colour
    # ----------------------
    def choose_colour(self):
        try:
            self.selected_colour = askcolor()
            self.color_hex = self.selected_colour[1]
            self.selected_colour = tuple(map(int, self.selected_colour[0]))
            self.currentColour.config(bg=self.color_hex)

            print("Colour", self.selected_colour, "selected")
        except:
            pass

    # ----------------------
    # Place sample
    # ----------------------
    def draw_px(self, event):
        if self.selected_colour == None:
            return

        print("Colour", self.selected_colour, "placed at", event.x, event.y)

        image_x = min(self.loaded_image.width - 1, event.x//self.transformations["scale"])
        image_y = min(self.loaded_image.height - 1, event.y//self.transformations["scale"])
        box = tuple(map(int, (max(image_x - 2, 0), min(image_x + 3, self.loaded_image.width), max(image_y - 2, 0), min(image_y + 3, self.loaded_image.height))))

        lab_colour = tuple(rgb2lab(np.asarray([[self.selected_colour]]).astype('B'))[0,0,:])

        self.LAB_image_data[box[2]:box[3], box[0]:box[1], 1] = lab_colour[1]
        self.LAB_image_data[box[2]:box[3], box[0]:box[1], 2] = lab_colour[2]

        self.image_data = 255*lab2rgb(self.LAB_image_data)

        self.loaded_image = Image.fromarray(self.image_data.astype('B'))
        self.render_image()

    # ----------------------
    # Zoom in
    # ----------------------
    def zoom_in(self):
        if self.loaded_image == None:
            return

        self.transformations["scale"] *= 2.0
        self.render_image()

    # ----------------------
    # Zoom out
    # ----------------------
    def zoom_out(self):
        if self.loaded_image == None:
            return

        self.transformations["scale"] *= 0.5
        self.render_image()

    # ----------------------
    # Render the image
    # ----------------------
    def render_image(self):
        if self.loaded_image == None:
            return

        scale = self.transformations["scale"]

        transformed_image = self.loaded_image.resize((int(self.loaded_image.width*scale), int(self.loaded_image.height*scale)))
        render = ImageTk.PhotoImage(transformed_image)
        self.image_label.configure(image=render)
        self.image_label.image = render

    # ----------------------
    # Call the NN
    # ----------------------
    def callmodel(self):
        new_img = torch.tensor(self.LAB_image_data).permute(2,0,1).unsqueeze(0).float()

        coloured_image = self.model(new_img)
        coloured_image = coloured_image.detach().squeeze().permute(1,2,0).numpy()
        plt.imshow(lab2rgb(coloured_image))
        plt.show()


if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

root = Tk()
root.geometry("800x450")

app = Window(root)
app.init_window()
root.mainloop()