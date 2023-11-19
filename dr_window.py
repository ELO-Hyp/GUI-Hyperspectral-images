#from tkinter import Tk, Label, StringVar, filedialog
from tkinter import ttk
import tkinter as tk
#from tkinter import messagebox
#import threading
#import time
import os

from DRmethod_pca import DRWindow_pca
from DRmethod_ica import DRWindow_ica
from DRmethod_lpp import DRWindow_lpp
from DRmethod_nmf import DRWindow_nmf
from DRmethod_osp import DRWindow_osp


class DRsubWindow:
    def __init__(self, window, window_title):

        self.window = window
        self.window.grab_set()  # Make it the main window.

        self.review_enroll_window = None
        self.settings_window_created = False

        self.window.title(window_title)
        self.window.minsize(300, 200)
        self.window.resizable(False, False)
        
        ttk.Label(text="").pack()  # just for spacing.

        # Create button for the Method 1: PCA
        ttk.Button(self.window, text="Principal Component Analysis", command=self.__get_dr_window, width=30).pack()
        
        # Create button for the Method 2: ICA
        ttk.Button(self.window, text="Independent Analysis Component", command=self.__get_dr_window_ica, width=30).pack()
        
        # Create button for the Method 4: NMF
        ttk.Button(self.window, text="Nonnegative Matrix Factorization", command=self.__get_dr_window_nmf, width=30).pack()
        

        # Create button for the Method 5: OSP
        ttk.Button(self.window, text="Orthogonal Subspace Projection", command=self.__get_dr_window_osp, width=30).pack()
        
        # Create button for the Method 3: LPP
        ttk.Button(self.window, text="Locality Preserving Projection", command=self.__get_dr_window_lpp, width=30).pack()


        ttk.Label(text="").pack()  # just for spacing.
#         #Add norway logo.
        self.window.iconbitmap(os.path.join("resources", 'elo-hyp_logo.ico'))



    def __get_dr_window(self):
        """Create a new top level window"""
        self.DRmethod = DRWindow_pca(tk.Toplevel(), "Principal Compoenent Analysis")
        
    def __get_dr_window_ica(self):
        """Create a new top level window"""
        self.DRmethod = DRWindow_ica(tk.Toplevel(), "Independent Analysis Component")
    
    def __get_dr_window_lpp(self):
        """Create a new top level window"""
        self.DRmethod = DRWindow_lpp(tk.Toplevel(), "Locality Preserving Projection")
    
    def __get_dr_window_osp(self):
        """Create a new top level window"""
        self.DRmethod = DRWindow_osp(tk.Toplevel(), "Orthogonal Subspace Projection")
        
    def __get_dr_window_nmf(self):
            """Create a new top level window"""
            self.DRmethod = DRWindow_nmf(tk.Toplevel(), "Nonnegative Matrix Factorization")
        
    