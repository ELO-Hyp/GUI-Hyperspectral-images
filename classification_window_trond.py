from tkinter import Tk, Label, StringVar, filedialog
from tkinter import ttk
import tkinter as tk
from tkinter import messagebox
import threading
import time
import os

from classification_Trond2023 import ClassificationTrond2023, ClassificationTrond2023_red_pca, ClassificationTrond2023_red_ica, ClassificationTrond2023_red_nmf, ClassificationTrond2023_red_osp, ClassificationTrond2023_red_lpp

class ClassificationWindowTrond:
    def __init__(self, window, window_title):

        self.window = window
        self.window.grab_set()  # Make it the main window.

        self.review_enroll_window = None
        self.settings_window_created = False

        self.window.title(window_title)
        self.window.minsize(300, 200)
        self.window.resizable(False, False)
        
        ttk.Label(text="").pack()  # just for spacing.
        

        # Create button for the Data 2
        ttk.Button(self.window, text="Trondheim scene 2023", command=self.__get_classification_window1, width=30).pack()
        
        # Create button for the Data reduction
        ttk.Button(self.window, text="Trondheim scene 2023 pca", command=self.__get_classification_window1_red, width=30).pack()
        
        # Create button for the Data reduction ica
        ttk.Button(self.window, text="Trondheim scene 2023 ica", command=self.__get_classification_window1_red_ica, width=30).pack()
        
        # Create button for the Data reduction nmf
        ttk.Button(self.window, text="Trondheim scene 2023 nmf", command=self.__get_classification_window1_red_nmf, width=30).pack()
        
        # Create button for the Data reduction nmf
        ttk.Button(self.window, text="Trondheim scene 2023 osp", command=self.__get_classification_window1_red_osp, width=30).pack()
        
        # Create button for the Data reduction nmf
        ttk.Button(self.window, text="Trondheim scene 2023 lpp", command=self.__get_classification_window1_red_lpp, width=30).pack()
        

        ttk.Label(text="").pack()  # just for spacing.
#         #Add norway logo.
        self.window.iconbitmap(os.path.join("Resources", 'elo-hyp_logo.ico'))


    def __get_classification_window1(self):
        """Create a new top level window"""
        self.classification_Trond2023 = ClassificationTrond2023(tk.Toplevel(), "SSVM-Classification")
        
    def __get_classification_window1_red(self):
        """Create a new top level window"""
        self.classification_Trond2023 = ClassificationTrond2023_red_pca(tk.Toplevel(), "SSVM-Classification")
        
    def __get_classification_window1_red_ica(self):
        """Create a new top level window"""
        self.classification_Trond2023 = ClassificationTrond2023_red_ica(tk.Toplevel(), "SSVM-Classification")
        
    def __get_classification_window1_red_nmf(self):
        """Create a new top level window"""
        self.classification_Trond2023 = ClassificationTrond2023_red_nmf(tk.Toplevel(), "SSVM-Classification")
        
    def __get_classification_window1_red_osp(self):
        """Create a new top level window"""
        self.classification_Trond2023 = ClassificationTrond2023_red_osp(tk.Toplevel(), "SSVM-Classification")
        
    def __get_classification_window1_red_lpp(self):
        """Create a new top level window"""
        self.classification_Trond2023 = ClassificationTrond2023_red_lpp(tk.Toplevel(), "SSVM-Classification")
        
