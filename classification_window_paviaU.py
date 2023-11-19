from tkinter import Tk, Label, StringVar, filedialog
from tkinter import ttk
import tkinter as tk
from tkinter import messagebox
import threading
import time
import os

from classification_PaviaU import ClassificationPaviaU, ClassificationPaviaU_red_pca, ClassificationPaviaU_red_ica, ClassificationPaviaU_red_nmf, ClassificationPaviaU_red_osp, ClassificationPaviaU_red_lpp

class ClassificationWindowPaviaU:
    def __init__(self, window, window_title):

        self.window = window
        self.window.grab_set()  # Make it the main window.

        self.review_enroll_window = None
        self.settings_window_created = False

        self.window.title(window_title)
        self.window.minsize(300, 200)
        self.window.resizable(False, False)
        
        ttk.Label(text="").pack()  # just for spacing.
        
        
        # Create button for the Data
        ttk.Button(self.window, text="PaviaU data", command=self.__get_classification_window4, width=30).pack()
        
        # Create button for the Data 
        ttk.Button(self.window, text="PaviaU data pca", command=self.__get_classification_window4_red_pca, width=30).pack()
        
        # Create button for the Data 
        ttk.Button(self.window, text="PaviaU data ica", command=self.__get_classification_window4_red_ica, width=30).pack()
        
        # Create button for the Data 
        ttk.Button(self.window, text="PaviaU data nmf", command=self.__get_classification_window4_red_nmf, width=30).pack()
        
        # Create button for the Data 
        ttk.Button(self.window, text="PaviaU data osp", command=self.__get_classification_window4_red_osp, width=30).pack()
        
        # Create button for the Data 
        ttk.Button(self.window, text="PaviaU data lpp", command=self.__get_classification_window4_red_lpp, width=30).pack()


        ttk.Label(text="").pack()  # just for spacing.
#         #Add norway logo.
        self.window.iconbitmap(os.path.join("Resources", 'elo-hyp_logo.ico'))

    
    def __get_classification_window4(self):
        """Create a new top level window"""
        self.classification_PaviaU = ClassificationPaviaU(tk.Toplevel(), "SSVM-Classification")
        
    def __get_classification_window4_red_pca(self):
        """Create a new top level window"""
        self.classification_PaviaU = ClassificationPaviaU_red_pca(tk.Toplevel(), "SSVM-Classification")
        
    def __get_classification_window4_red_ica(self):
        """Create a new top level window"""
        self.classification_PaviaU = ClassificationPaviaU_red_ica(tk.Toplevel(), "SSVM-Classification")
        
    def __get_classification_window4_red_nmf(self):
        """Create a new top level window"""
        self.classification_PaviaU = ClassificationPaviaU_red_nmf(tk.Toplevel(), "SSVM-Classification")
        
    def __get_classification_window4_red_osp(self):
        """Create a new top level window"""
        self.classification_PaviaU = ClassificationPaviaU_red_osp(tk.Toplevel(), "SSVM-Classification")
        
    def __get_classification_window4_red_lpp(self):
        """Create a new top level window"""
        self.classification_PaviaU = ClassificationPaviaU_red_lpp(tk.Toplevel(), "SSVM-Classification")
