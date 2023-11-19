from tkinter import Tk, Label, StringVar, filedialog
from tkinter import ttk
import tkinter as tk
from tkinter import messagebox
import threading
import time
import os

from classification_Samson import ClassificationSamson, ClassificationSamson_red_pca, ClassificationSamson_red_ica, ClassificationSamson_red_nmf, ClassificationSamson_red_osp, ClassificationSamson_red_lpp

class ClassificationWindowSamson:
    def __init__(self, window, window_title):

        self.window = window
        self.window.grab_set()  # Make it the main window.

        self.review_enroll_window = None
        self.settings_window_created = False

        self.window.title(window_title)
        self.window.minsize(300, 200)
        self.window.resizable(False, False)
        
        ttk.Label(text="").pack()  # just for spacing.
        
        
        # Create button for the Data 4
        ttk.Button(self.window, text="Samson data", command=self.__get_classification_window3, width=30).pack()
        
        # Create button for the Data reduction
        ttk.Button(self.window, text="Samson data pca", command=self.__get_classification_window3_red, width=30).pack()
        
        # Create button for the Data reduction ica
        ttk.Button(self.window, text="Samson data ica", command=self.__get_classification_window3_red_ica, width=30).pack()
        
        # Create button for the Data reduction ica
        ttk.Button(self.window, text="Samson data nmf", command=self.__get_classification_window3_red_nmf, width=30).pack()
        
        # Create button for the Data reduction ica
        ttk.Button(self.window, text="Samson data osp", command=self.__get_classification_window3_red_osp, width=30).pack()
        
        # Create button for the Data reduction ica
        ttk.Button(self.window, text="Samson data lpp", command=self.__get_classification_window3_red_lpp, width=30).pack()


        ttk.Label(text="").pack()  # just for spacing.
#         #Add norway logo.
        self.window.iconbitmap(os.path.join("resources", 'elo-hyp_logo.ico'))

        
    def __get_classification_window3(self):
        """Create a new top level window"""
        self.classification_Samson = ClassificationSamson(tk.Toplevel(), "SSVM-Classification")
        
    def __get_classification_window3_red(self):
        """Create a new top level window"""
        self.classification_Samson = ClassificationSamson_red_pca(tk.Toplevel(), "SSVM-Classification")
        
    def __get_classification_window3_red_ica(self):
        """Create a new top level window"""
        self.classification_Samson = ClassificationSamson_red_ica(tk.Toplevel(), "SSVM-Classification")
        
    def __get_classification_window3_red_nmf(self):
        """Create a new top level window"""
        self.classification_Samson = ClassificationSamson_red_nmf(tk.Toplevel(), "SSVM-Classification")
        
    def __get_classification_window3_red_osp(self):
        """Create a new top level window"""
        self.classification_Samson = ClassificationSamson_red_osp(tk.Toplevel(), "SSVM-Classification")
        
    def __get_classification_window3_red_lpp(self):
        """Create a new top level window"""
        self.classification_Samson = ClassificationSamson_red_lpp(tk.Toplevel(), "SSVM-Classification")
        