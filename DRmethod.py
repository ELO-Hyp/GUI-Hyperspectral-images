from tkinter import Tk, Label, StringVar, filedialog
from tkinter import ttk
from tkinter import messagebox
import threading
import time

import ssvm
import scipy.io
from sklearn import preprocessing
import numpy as np
from functools import partial
import os
import matplotlib.pyplot as plt



class DRWindow:
    def __init__(self, window, window_title):

        self.window = window
        self.window.grab_set()  # Make it the main window.

        self.review_enroll_window = None
        self.settings_window_created = False

        self.window.title(window_title)
        self.window.minsize(750, 100)
        self.window.resizable(False, False)
        
        ####################################################################################

        # Folder output folder.
        self.label_saving_folder = Label(window, text="Saving folder:", font='Arial 12')
        self.label_saving_folder.place(relx=0.05, rely=0.05)
        self.label_saving_folder_path = Label(window, text="")
        self.label_saving_folder_path.place(relx=0.35, rely=0.06)
        fct_folder_save = partial(self.select_folder, self.label_saving_folder_path)
        self.button_saving_folder = ttk.Button(window, text="Browse folder", command=fct_folder_save)
        self.button_saving_folder.place(relx=0.05, rely=0.2)

        self.counter = 0
        self.processing_thread = None

        self.label_processing = Label(window, text="",  fg="black")
        self.label_processing.place(relx=0.35, rely=0.5)

        self.button_start_processing = ttk.Button(window, text="Start process",
                                                  command=self.__start_processing)
        self.button_start_processing.place(relx=0.05, rely=0.6)
        self.stop_thread = False
        self.__shown_text = ""
        self.__num_of_processing_images = 0
        self.__update()
        
        ###################################################################################
        
        self.window.iconbitmap(os.path.join("resources", 'elo-hyp_logo.ico'))

    def select_folder(self, storing_label):
        filename = filedialog.askdirectory(title="Select a Folder")
        storing_label.configure(text=filename)

    def __update(self):
        if self.processing_thread is None:
            self.label_processing.configure(text=self.__shown_text)
        elif self.processing_thread.is_alive():
            self.label_processing.configure(text=f"Processing: {self.counter}/{self.__num_of_processing_images}.")
        else:
            self.label_processing.configure(text=self.__shown_text)

        self.window.after(1000, self.__update)

#     def __process(self, input_dir: str, output_dir: str):
    def __process(self, output_dir: str):
        try:
                    
################ load weights  ###################
            ssvm2 = ssvm.SSVM()
            ssvm2.load('post_process_data/Tron.json')
                    
############## load data #####################
            Tron = np.fromfile('Trondheim_data/trondheim_2023-03-27_0939Z.bip', dtype=np.uint16)
            Tron = Tron.reshape((956,228,360)).T
            Tron.shape # (bands X hight x width)
            Tron[1,:,:].shape
            Tron2 = np.zeros((228*956,360))
            for i in range(360):
                Tron2[:,i] = np.array(Tron[i,:,:]).flatten()
            y = np.load('Trondheim_data/trondheim_2023-03-27_0939Z-labels.npy')
            X = preprocessing.scale(Tron2, axis=0)                    # Normalization

            y[y == 0] = 2
            y[y == 9] = 4                      # there are only 12 pixels with class 9 so we merge them with class 4
                    
                    # full data accuracy
            x_full_guess = ssvm2.predict(X)
            correct_prediction = x_full_guess - y
            N = len(y)
            print('Total number of data for testing =', N)
            print('correctly predicted =',len(correct_prediction[np.where(correct_prediction == 0)]))
            Accuracy = len(correct_prediction[np.where(correct_prediction == 0)])*100/N
            print('Accuracy (in %) =', Accuracy)
                    
####################################### to save the output  ###########################
            path_to_save = os.path.join(output_dir, "output_details.txt")
            file1 = open(path_to_save, "w")
            toFile = "Accuracy (in %) = "
            file1.write('{a} {b}'.format(a =toFile, b = Accuracy))
            file1.close()
            I = [2, 4, 5, 6]
            for i in range(4):
                plt.plot(ssvm2._w()[I[i]],label='Class %s' %I[i])
    
            plt.legend(title='sparsity')
            plt.xlabel("Number of bands")
            plt.ylabel("Weights")
            plt.savefig(output_dir+'/SVM_weights_Trondheim.pdf')

            self.__shown_text = f"Results saved at: {output_dir}\n Accuracy (in %) = {Accuracy}"
            self.button_start_processing.config(state='normal')
            self.counter = 0
        except Exception as ex:
            print(ex)
            self.__shown_text = "An exception occurred!"

    def __start_processing(self):

        save_dir = self.label_saving_folder_path["text"]
        if save_dir is None or save_dir == "":
            messagebox.showerror("Error", "The saving folder must be selected!")
            return

        self.button_start_processing.config(state='disabled')

        self.processing_thread = threading.Thread(target=self.__process,
                                                  args=(save_dir, ), daemon=True)
        self.processing_thread.start()