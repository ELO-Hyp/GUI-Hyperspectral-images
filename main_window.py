import os
import tkinter as tk
from tkinter import ttk

import psutil
from PIL import ImageTk, Image

from classification_window_trond import ClassificationWindowTrond
from classification_window_paviaU import ClassificationWindowPaviaU
from classification_window_samson import ClassificationWindowSamson
from dr_window import DRsubWindow



class MainWindow:
    def __init__(self, root, name):
        self.root = root
        self.root.title(name)

        # Set theme.
        self.root.tk.call("source", os.path.join("Resources", "UI", "sun-valley.tcl"))
        self.root.tk.call("set_theme", "light")

        # Set the geometry of tkinter frame
        self.root.geometry("350x450")
        # Add image logo.
        image1 = Image.open(os.path.join("Resources", "elo-hyp_logo.png")).resize((100, 100))
        test = ImageTk.PhotoImage(image1)

        label1 = tk.Label(image=test)
        label1.image = test
        # Position image
        label1.pack()

        ttk.Label(text="").pack()  # just for spacing.

        # Create button for the DR
        ttk.Button(self.root, text="Dimensionality-Reduction", command=self.__get_dr_window, width=30).pack()

        # Create button for the Classification Trondheim 2023 data
        ttk.Button(self.root, text="Classification (Trondheim 2023)", command=self.__get_classification_window1, width=30).pack()
        
        # Create button for the Classification PaviaU data
        ttk.Button(self.root, text="Classification (PaviaU)", command=self.__get_classification_window2, width=30).pack()
        
        # Create button for the Classification Samson data
        ttk.Button(self.root, text="Classification (Samson)", command=self.__get_classification_window3, width=30).pack()


        ttk.Label(text="").pack()  # just for spacing.
        # Add norway logo.
        image_2 = Image.open(os.path.join("Resources", "norway_grants_logo.png")).resize((50, 50))
        test = ImageTk.PhotoImage(image_2)

        label_2 = tk.Label(image=test)
        label_2.image = test
        # Position image
        label_2.pack()

        label_3 = ttk.Label(text="\n***The research leading to this application has received \nfunding from"
                                 " the NO Grants 2014-2021, under project\nELO-Hyp contract no. 24/2020.")
        label_3.pack()
        self.root.iconbitmap(os.path.join("Resources", 'elo-hyp_logo.ico'))
        self.root.mainloop()


    def __get_classification_window1(self):
        """Create a new top level window"""
        self.classification_window_trond = ClassificationWindowTrond(tk.Toplevel(), "Hyperspectral Data")
        
    def __get_classification_window2(self):
        """Create a new top level window"""
        self.classification_window_paviaU = ClassificationWindowPaviaU(tk.Toplevel(), "Hyperspectral Data")
        
    def __get_classification_window3(self):
        """Create a new top level window"""
        self.classification_window_samson = ClassificationWindowSamson(tk.Toplevel(), "Hyperspectral Data")
        
    def __get_dr_window(self):
        """Create a new top level window"""
        self.dr_window = DRsubWindow(tk.Toplevel(), "Methods")


