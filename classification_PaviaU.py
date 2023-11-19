from tkinter import Tk, Label, StringVar, filedialog
from tkinter import ttk
from tkinter import messagebox
import threading
import time

import ssvm
from scipy import io as sio
import scipy.io
from sklearn import preprocessing
import numpy as np
from functools import partial
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')


class ClassificationPaviaU:
    def __init__(self, window, window_title):

        self.window = window
        self.window.grab_set()  # Make it the main window.

        self.review_enroll_window = None
        self.settings_window_created = False

        self.window.title(window_title)
        self.window.minsize(750, 150)
        self.window.resizable(False, False)
        
        ####################################################################################

       # Folder input folder.
        self.label_folder_imgs = Label(window, text="Processing data:", fg="black", font='Arial 12')
        self.label_folder_imgs.place(relx=0.05, rely=0.05)
        self.label_folder_imgs_path = Label(window, text="", fg="black")
        self.label_folder_imgs_path.place(relx=0.35, rely=0.06)
        fct_folder_imgs = partial(self.select_file, self.label_folder_imgs_path)
        self.button_folder = ttk.Button(window, text="Browse file", command=fct_folder_imgs)
        self.button_folder.place(relx=0.05, rely=0.2)

        
        # Folder output folder.
        self.label_saving_folder = Label(window, text="Saving folder:", font='Arial 12')
        self.label_saving_folder.place(relx=0.05, rely=0.38)
        self.label_saving_folder_path = Label(window, text="")
        self.label_saving_folder_path.place(relx=0.35, rely=0.38)
        fct_folder_save = partial(self.select_folder, self.label_saving_folder_path)
        self.button_saving_folder = ttk.Button(window, text="Browse folder", command=fct_folder_save)
        self.button_saving_folder.place(relx=0.05, rely=0.53)

        self.counter = 0
        self.processing_thread = None

        self.label_processing = Label(window, text="",  fg="black")
        self.label_processing.place(relx=0.35, rely=0.7)

        self.button_start_processing = ttk.Button(window, text="Start process",
                                                  command=self.__start_processing)
        self.button_start_processing.place(relx=0.05, rely=0.75)
        self.stop_thread = False
        self.__shown_text = ""
        self.__num_of_processing_images = 0
        self.__update()
        
        ###################################################################################
        
        self.window.iconbitmap(os.path.join("Resources", 'elo-hyp_logo.ico'))

    def select_folder(self, storing_label):
        filename = filedialog.askdirectory(title="Select a Folder")
        storing_label.configure(text=filename)
        
    def select_file(self, storing_label):
        filename = filedialog.askopenfilename(title="Select a File")
        storing_label.configure(text=filename)

    def __update(self):
        if self.processing_thread is None:
            self.label_processing.configure(text=self.__shown_text)
        elif self.processing_thread.is_alive():
            self.label_processing.configure(text=f"Processing: {self.counter}/{self.__num_of_processing_images}.")
        else:
            self.label_processing.configure(text=self.__shown_text)

        self.window.after(1000, self.__update)

    def __process(self, input_dir: str, output_dir: str):
        try:
                    
################ load weights  ###################
            ssvm2 = ssvm.SSVM()
            ssvm2.load('post_process_data/PaviaU.json')
                    
############## load data #####################
            data = sio.loadmat(input_dir)['paviaU']
            labels = sio.loadmat('data/PaviaU_gt.mat')['paviaU_gt']
            X = np.zeros((610*340,103))
            for i in range(103):
                X[:,i] = np.array(data[:,:,i]).flatten()
            y = np.array(labels).flatten()
            X = preprocessing.scale(X, axis=0)                    # Normalization
            I = [0, 1, 2,3,4,5,6,7,8,9]
            
                    # full data accuracy
            x_full_guess = ssvm2.predict(X)
            correct_prediction = x_full_guess - y
            N = len(y)
#             print('Total number of data for testing =', N)
            cp = len(correct_prediction[np.where(correct_prediction == 0)])
            Accuracy = len(correct_prediction[np.where(correct_prediction == 0)])*100/N
            print('Accuracy (in %) =', Accuracy)
            guess = []
            for i in range(10):
                L = N - sum(y == I[i]) + len(np.intersect1d(np.where(x_full_guess == I[i]),np.where(y == I[i])))
                guess = np.append(guess, L*100/N)
                    
####################################### to save the output  ###########################
            ### .txt file
            path_to_save = os.path.join(output_dir, "output_details_PaviaU.txt")
            file1 = open(path_to_save, "w")
            file1.write("Overall Accuracy----\n")
            toFile = "Total number of data for testing ="
            toFile1 = "Correctly predicted = "
            toFile2 = "Overall Accuracy (in %) = "
            file1.write('\n{a} {b} \n{c} {d} \n{e} {f}\n'.format(a=toFile, b=N, c=toFile1, d=cp, e=toFile2, f=Accuracy))
            file1.write("\n----Class-wise Accuracy and Average Accuracy----\n")
            for i in range(10):
                F = "Accuracy of class %s" %I[i]
                file1.write('{a} (in %) = {b}\n'.format(a=F, b=guess[i]))
            File = "Average Accuracy (in %) = "
            file1.write( '{a} {b}'.format(a=File, b= np.mean(guess)))
            file1.close()
            ### figure
            plt.close("all")
            for i in range(10):
                plt.plot(ssvm2._w()[I[i]],label='Class %s' %I[i])
    
            lgd = plt.legend(title='sparsity', loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xlabel("Number of bands")
            plt.ylabel("Weights")
            plt.savefig(output_dir+'/SVM_weights_PaviaU.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

            self.__shown_text = f"Results saved at: {output_dir}\n Overall Accuracy (in %) = {Accuracy}"
            self.button_start_processing.config(state='normal')
            self.counter = 0
        except Exception as ex:
            print(ex)
            self.__shown_text = "An exception occurred!"

    def __start_processing(self):

        input_dir = self.label_folder_imgs_path["text"]
        if input_dir is None or input_dir == "":
            messagebox.showerror("Error", "The processing folder must be selected!")
            return

        save_dir = self.label_saving_folder_path["text"]
        if save_dir is None or save_dir == "":
            messagebox.showerror("Error", "The saving folder must be selected!")
            return

        self.button_start_processing.config(state='disabled')

        self.processing_thread = threading.Thread(target=self.__process,
                                                  args=(input_dir, save_dir, ), daemon=True)
        self.processing_thread.start()
        
        
class ClassificationPaviaU_red_pca:
    def __init__(self, window, window_title):

        self.window = window
        self.window.grab_set()  # Make it the main window.

        self.review_enroll_window = None
        self.settings_window_created = False

        self.window.title(window_title)
        self.window.minsize(750, 150)
        self.window.resizable(False, False)
        
        ####################################################################################

        # Folder input folder.
        self.label_folder_imgs = Label(window, text="Processing data:", fg="black", font='Arial 12')
        self.label_folder_imgs.place(relx=0.05, rely=0.05)
        self.label_folder_imgs_path = Label(window, text="", fg="black")
        self.label_folder_imgs_path.place(relx=0.35, rely=0.06)
        fct_folder_imgs = partial(self.select_file, self.label_folder_imgs_path)
        self.button_folder = ttk.Button(window, text="Browse file", command=fct_folder_imgs)
        self.button_folder.place(relx=0.05, rely=0.2)

        
        # Folder output folder.
        self.label_saving_folder = Label(window, text="Saving folder:", font='Arial 12')
        self.label_saving_folder.place(relx=0.05, rely=0.38)
        self.label_saving_folder_path = Label(window, text="")
        self.label_saving_folder_path.place(relx=0.35, rely=0.38)
        fct_folder_save = partial(self.select_folder, self.label_saving_folder_path)
        self.button_saving_folder = ttk.Button(window, text="Browse folder", command=fct_folder_save)
        self.button_saving_folder.place(relx=0.05, rely=0.53)

        self.counter = 0
        self.processing_thread = None

        self.label_processing = Label(window, text="",  fg="black")
        self.label_processing.place(relx=0.35, rely=0.7)

        self.button_start_processing = ttk.Button(window, text="Start process",
                                                  command=self.__start_processing)
        self.button_start_processing.place(relx=0.05, rely=0.75)
        self.stop_thread = False
        self.__shown_text = ""
        self.__num_of_processing_images = 0
        self.__update()
        
        ###################################################################################
        
        self.window.iconbitmap(os.path.join("Resources", 'elo-hyp_logo.ico'))

    def select_folder(self, storing_label):
        filename = filedialog.askdirectory(title="Select a Folder")
        storing_label.configure(text=filename)
        
    def select_file(self, storing_label):
        filename = filedialog.askopenfilename(title="Select a File")
        storing_label.configure(text=filename)

    def __update(self):
        if self.processing_thread is None:
            self.label_processing.configure(text=self.__shown_text)
        elif self.processing_thread.is_alive():
            self.label_processing.configure(text=f"Processing: {self.counter}/{self.__num_of_processing_images}.")
        else:
            self.label_processing.configure(text=self.__shown_text)

        self.window.after(1000, self.__update)

    def __process(self, input_dir: str, output_dir: str):
#     def __process(self, output_dir: str):
        try:
                    
################ load weights  ###################
            ssvm2 = ssvm.SSVM()
            ssvm2.load('post_process_data/PaviaU_red_pca.json')
                    
############## load data #####################
            data = scipy.io.loadmat(input_dir)['hyp_img_red']
            labels = sio.loadmat('data/PaviaU_gt.mat')['paviaU_gt']
            X = data.T
            y = np.array(labels).flatten()
            X = preprocessing.scale(X, axis=0)                    # Normalization
            I = [0, 1, 2,3,4,5,6,7,8,9]
            
                    # full data accuracy
            x_full_guess = ssvm2.predict(X)
            correct_prediction = x_full_guess - y
            N = len(y)
#             print('Total number of data for testing =', N)
            cp = len(correct_prediction[np.where(correct_prediction == 0)])
            Accuracy = len(correct_prediction[np.where(correct_prediction == 0)])*100/N
            print('Accuracy (in %) =', Accuracy)
            guess = []
            for i in range(10):
                L = N - sum(y == I[i]) + len(np.intersect1d(np.where(x_full_guess == I[i]),np.where(y == I[i])))
                guess = np.append(guess, L*100/N)
                    
####################################### to save the output  ###########################
            ### .txt file
            path_to_save = os.path.join(output_dir, "output_details_PaviaU_pca.txt")
            file1 = open(path_to_save, "w")
            file1.write("Overall Accuracy----\n")
            toFile = "Total number of data for testing ="
            toFile1 = "Correctly predicted = "
            toFile2 = "Overall Accuracy (in %) = "
            file1.write('\n{a} {b} \n{c} {d} \n{e} {f}\n'.format(a=toFile, b=N, c=toFile1, d=cp, e=toFile2, f=Accuracy))
            file1.write("\n----Class-wise Accuracy and Average Accuracy----\n")
            for i in range(10):
                F = "Accuracy of class %s" %I[i]
                file1.write('{a} (in %) = {b}\n'.format(a=F, b=guess[i]))
            File = "Average Accuracy (in %) = "
            file1.write( '{a} {b}'.format(a=File, b= np.mean(guess)))
            file1.close()
            ### figure
            plt.close("all")
            for i in range(10):
                plt.plot(ssvm2._w()[I[i]],label='Class %s' %I[i])
    
            lgd = plt.legend(title='sparsity', loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xlabel("Number of bands")
            plt.ylabel("Weights")
            plt.savefig(output_dir+'/SVM_weights_PaviaU_pca.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

            self.__shown_text = f"Results saved at: {output_dir}\n Overall Accuracy (in %) = {Accuracy}"
            self.button_start_processing.config(state='normal')
            self.counter = 0
        except Exception as ex:
            print(ex)
            self.__shown_text = "An exception occurred!"

    def __start_processing(self):
        
        input_dir = self.label_folder_imgs_path["text"]
        if input_dir is None or input_dir == "":
            messagebox.showerror("Error", "The processing folder must be selected!")
            return

        save_dir = self.label_saving_folder_path["text"]
        if save_dir is None or save_dir == "":
            messagebox.showerror("Error", "The saving folder must be selected!")
            return

        self.button_start_processing.config(state='disabled')

        self.processing_thread = threading.Thread(target=self.__process,
                                                  args=(input_dir, save_dir, ), daemon=True)
        self.processing_thread.start()
        
        
class ClassificationPaviaU_red_ica:
    def __init__(self, window, window_title):

        self.window = window
        self.window.grab_set()  # Make it the main window.

        self.review_enroll_window = None
        self.settings_window_created = False

        self.window.title(window_title)
        self.window.minsize(750, 150)
        self.window.resizable(False, False)
        
        ####################################################################################

        # Folder input folder.
        self.label_folder_imgs = Label(window, text="Processing data:", fg="black", font='Arial 12')
        self.label_folder_imgs.place(relx=0.05, rely=0.05)
        self.label_folder_imgs_path = Label(window, text="", fg="black")
        self.label_folder_imgs_path.place(relx=0.35, rely=0.06)
        fct_folder_imgs = partial(self.select_file, self.label_folder_imgs_path)
        self.button_folder = ttk.Button(window, text="Browse file", command=fct_folder_imgs)
        self.button_folder.place(relx=0.05, rely=0.2)

        
        # Folder output folder.
        self.label_saving_folder = Label(window, text="Saving folder:", font='Arial 12')
        self.label_saving_folder.place(relx=0.05, rely=0.38)
        self.label_saving_folder_path = Label(window, text="")
        self.label_saving_folder_path.place(relx=0.35, rely=0.38)
        fct_folder_save = partial(self.select_folder, self.label_saving_folder_path)
        self.button_saving_folder = ttk.Button(window, text="Browse folder", command=fct_folder_save)
        self.button_saving_folder.place(relx=0.05, rely=0.53)

        self.counter = 0
        self.processing_thread = None

        self.label_processing = Label(window, text="",  fg="black")
        self.label_processing.place(relx=0.35, rely=0.7)

        self.button_start_processing = ttk.Button(window, text="Start process",
                                                  command=self.__start_processing)
        self.button_start_processing.place(relx=0.05, rely=0.75)
        self.stop_thread = False
        self.__shown_text = ""
        self.__num_of_processing_images = 0
        self.__update()
        
        ###################################################################################
        
        self.window.iconbitmap(os.path.join("Resources", 'elo-hyp_logo.ico'))

    def select_folder(self, storing_label):
        filename = filedialog.askdirectory(title="Select a Folder")
        storing_label.configure(text=filename)
        
    def select_file(self, storing_label):
        filename = filedialog.askopenfilename(title="Select a File")
        storing_label.configure(text=filename)

    def __update(self):
        if self.processing_thread is None:
            self.label_processing.configure(text=self.__shown_text)
        elif self.processing_thread.is_alive():
            self.label_processing.configure(text=f"Processing: {self.counter}/{self.__num_of_processing_images}.")
        else:
            self.label_processing.configure(text=self.__shown_text)

        self.window.after(1000, self.__update)

    def __process(self, input_dir: str, output_dir: str):
#     def __process(self, output_dir: str):
        try:
                    
################ load weights  ###################
            ssvm2 = ssvm.SSVM()
            ssvm2.load('post_process_data/PaviaU_red_ica.json')
                    
############## load data #####################
            data = scipy.io.loadmat(input_dir)['hyp_img_red']
            labels = sio.loadmat('data/PaviaU_gt.mat')['paviaU_gt']
            X = data.T
            y = np.array(labels).flatten()
            X = preprocessing.scale(X, axis=0)                    # Normalization
            I = [0, 1, 2,3,4,5,6,7,8,9]
            
                    # full data accuracy
            x_full_guess = ssvm2.predict(X)
            correct_prediction = x_full_guess - y
            N = len(y)
#             print('Total number of data for testing =', N)
            cp = len(correct_prediction[np.where(correct_prediction == 0)])
            Accuracy = len(correct_prediction[np.where(correct_prediction == 0)])*100/N
            print('Accuracy (in %) =', Accuracy)
            guess = []
            for i in range(10):
                L = N - sum(y == I[i]) + len(np.intersect1d(np.where(x_full_guess == I[i]),np.where(y == I[i])))
                guess = np.append(guess, L*100/N)
                    
####################################### to save the output  ###########################
            ### .txt file
            path_to_save = os.path.join(output_dir, "output_details_PaviaU_ica.txt")
            file1 = open(path_to_save, "w")
            file1.write("Overall Accuracy----\n")
            toFile = "Total number of data for testing ="
            toFile1 = "Correctly predicted = "
            toFile2 = "Overall Accuracy (in %) = "
            file1.write('\n{a} {b} \n{c} {d} \n{e} {f}\n'.format(a=toFile, b=N, c=toFile1, d=cp, e=toFile2, f=Accuracy))
            file1.write("\n----Class-wise Accuracy and Average Accuracy----\n")
            for i in range(10):
                F = "Accuracy of class %s" %I[i]
                file1.write('{a} (in %) = {b}\n'.format(a=F, b=guess[i]))
            File = "Average Accuracy (in %) = "
            file1.write( '{a} {b}'.format(a=File, b= np.mean(guess)))
            file1.close()
            ### figure
            plt.close("all")
            for i in range(10):
                plt.plot(ssvm2._w()[I[i]],label='Class %s' %I[i])
    
            lgd = plt.legend(title='sparsity', loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xlabel("Number of bands")
            plt.ylabel("Weights")
            plt.savefig(output_dir+'/SVM_weights_PaviaU_ica.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

            self.__shown_text = f"Results saved at: {output_dir}\n Overall Accuracy (in %) = {Accuracy}"
            self.button_start_processing.config(state='normal')
            self.counter = 0
        except Exception as ex:
            print(ex)
            self.__shown_text = "An exception occurred!"

    def __start_processing(self):
        
        input_dir = self.label_folder_imgs_path["text"]
        if input_dir is None or input_dir == "":
            messagebox.showerror("Error", "The processing folder must be selected!")
            return

        save_dir = self.label_saving_folder_path["text"]
        if save_dir is None or save_dir == "":
            messagebox.showerror("Error", "The saving folder must be selected!")
            return

        self.button_start_processing.config(state='disabled')

        self.processing_thread = threading.Thread(target=self.__process,
                                                  args=(input_dir, save_dir, ), daemon=True)
        self.processing_thread.start()
        
        
class ClassificationPaviaU_red_nmf:
    def __init__(self, window, window_title):

        self.window = window
        self.window.grab_set()  # Make it the main window.

        self.review_enroll_window = None
        self.settings_window_created = False

        self.window.title(window_title)
        self.window.minsize(750, 150)
        self.window.resizable(False, False)
        
        ####################################################################################

        # Folder input folder.
        self.label_folder_imgs = Label(window, text="Processing data:", fg="black", font='Arial 12')
        self.label_folder_imgs.place(relx=0.05, rely=0.05)
        self.label_folder_imgs_path = Label(window, text="", fg="black")
        self.label_folder_imgs_path.place(relx=0.35, rely=0.06)
        fct_folder_imgs = partial(self.select_file, self.label_folder_imgs_path)
        self.button_folder = ttk.Button(window, text="Browse file", command=fct_folder_imgs)
        self.button_folder.place(relx=0.05, rely=0.2)

        
        # Folder output folder.
        self.label_saving_folder = Label(window, text="Saving folder:", font='Arial 12')
        self.label_saving_folder.place(relx=0.05, rely=0.38)
        self.label_saving_folder_path = Label(window, text="")
        self.label_saving_folder_path.place(relx=0.35, rely=0.38)
        fct_folder_save = partial(self.select_folder, self.label_saving_folder_path)
        self.button_saving_folder = ttk.Button(window, text="Browse folder", command=fct_folder_save)
        self.button_saving_folder.place(relx=0.05, rely=0.53)

        self.counter = 0
        self.processing_thread = None

        self.label_processing = Label(window, text="",  fg="black")
        self.label_processing.place(relx=0.35, rely=0.7)

        self.button_start_processing = ttk.Button(window, text="Start process",
                                                  command=self.__start_processing)
        self.button_start_processing.place(relx=0.05, rely=0.75)
        self.stop_thread = False
        self.__shown_text = ""
        self.__num_of_processing_images = 0
        self.__update()
        
        ###################################################################################
        
        self.window.iconbitmap(os.path.join("Resources", 'elo-hyp_logo.ico'))

    def select_folder(self, storing_label):
        filename = filedialog.askdirectory(title="Select a Folder")
        storing_label.configure(text=filename)
        
    def select_file(self, storing_label):
        filename = filedialog.askopenfilename(title="Select a File")
        storing_label.configure(text=filename)

    def __update(self):
        if self.processing_thread is None:
            self.label_processing.configure(text=self.__shown_text)
        elif self.processing_thread.is_alive():
            self.label_processing.configure(text=f"Processing: {self.counter}/{self.__num_of_processing_images}.")
        else:
            self.label_processing.configure(text=self.__shown_text)

        self.window.after(1000, self.__update)

    def __process(self, input_dir: str, output_dir: str):
#     def __process(self, output_dir: str):
        try:
                    
################ load weights  ###################
            ssvm2 = ssvm.SSVM()
            ssvm2.load('post_process_data/PaviaU_red_nmf.json')
                    
############## load data #####################
            data = scipy.io.loadmat(input_dir)['hyp_img_red']
            labels = sio.loadmat('data/PaviaU_gt.mat')['paviaU_gt']
            X = data.T
            y = np.array(labels).flatten()
            X = preprocessing.scale(X, axis=0)                    # Normalization
            I = [0, 1, 2,3,4,5,6,7,8,9]
            
                    # full data accuracy
            x_full_guess = ssvm2.predict(X)
            correct_prediction = x_full_guess - y
            N = len(y)
#             print('Total number of data for testing =', N)
            cp = len(correct_prediction[np.where(correct_prediction == 0)])
            Accuracy = len(correct_prediction[np.where(correct_prediction == 0)])*100/N
            print('Accuracy (in %) =', Accuracy)
            guess = []
            for i in range(10):
                L = N - sum(y == I[i]) + len(np.intersect1d(np.where(x_full_guess == I[i]),np.where(y == I[i])))
                guess = np.append(guess, L*100/N)
                    
####################################### to save the output  ###########################
            ### .txt file
            path_to_save = os.path.join(output_dir, "output_details_PaviaU_nmf.txt")
            file1 = open(path_to_save, "w")
            file1.write("Overall Accuracy----\n")
            toFile = "Total number of data for testing ="
            toFile1 = "Correctly predicted = "
            toFile2 = "Overall Accuracy (in %) = "
            file1.write('\n{a} {b} \n{c} {d} \n{e} {f}\n'.format(a=toFile, b=N, c=toFile1, d=cp, e=toFile2, f=Accuracy))
            file1.write("\n----Class-wise Accuracy and Average Accuracy----\n")
            for i in range(10):
                F = "Accuracy of class %s" %I[i]
                file1.write('{a} (in %) = {b}\n'.format(a=F, b=guess[i]))
            File = "Average Accuracy (in %) = "
            file1.write( '{a} {b}'.format(a=File, b= np.mean(guess)))
            file1.close()
            ### figure
            plt.close("all")
            for i in range(10):
                plt.plot(ssvm2._w()[I[i]],label='Class %s' %I[i])
    
            lgd = plt.legend(title='sparsity', loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xlabel("Number of bands")
            plt.ylabel("Weights")
            plt.savefig(output_dir+'/SVM_weights_PaviaU_nmf.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

            self.__shown_text = f"Results saved at: {output_dir}\n Overall Accuracy (in %) = {Accuracy}"
            self.button_start_processing.config(state='normal')
            self.counter = 0
        except Exception as ex:
            print(ex)
            self.__shown_text = "An exception occurred!"

    def __start_processing(self):
        
        input_dir = self.label_folder_imgs_path["text"]
        if input_dir is None or input_dir == "":
            messagebox.showerror("Error", "The processing folder must be selected!")
            return

        save_dir = self.label_saving_folder_path["text"]
        if save_dir is None or save_dir == "":
            messagebox.showerror("Error", "The saving folder must be selected!")
            return

        self.button_start_processing.config(state='disabled')

        self.processing_thread = threading.Thread(target=self.__process,
                                                  args=(input_dir, save_dir, ), daemon=True)
        self.processing_thread.start()
        
        
class ClassificationPaviaU_red_osp:
    def __init__(self, window, window_title):

        self.window = window
        self.window.grab_set()  # Make it the main window.

        self.review_enroll_window = None
        self.settings_window_created = False

        self.window.title(window_title)
        self.window.minsize(750, 150)
        self.window.resizable(False, False)
        
        ####################################################################################

        # Folder input folder.
        self.label_folder_imgs = Label(window, text="Processing data:", fg="black", font='Arial 12')
        self.label_folder_imgs.place(relx=0.05, rely=0.05)
        self.label_folder_imgs_path = Label(window, text="", fg="black")
        self.label_folder_imgs_path.place(relx=0.35, rely=0.06)
        fct_folder_imgs = partial(self.select_file, self.label_folder_imgs_path)
        self.button_folder = ttk.Button(window, text="Browse file", command=fct_folder_imgs)
        self.button_folder.place(relx=0.05, rely=0.2)

        
        # Folder output folder.
        self.label_saving_folder = Label(window, text="Saving folder:", font='Arial 12')
        self.label_saving_folder.place(relx=0.05, rely=0.38)
        self.label_saving_folder_path = Label(window, text="")
        self.label_saving_folder_path.place(relx=0.35, rely=0.38)
        fct_folder_save = partial(self.select_folder, self.label_saving_folder_path)
        self.button_saving_folder = ttk.Button(window, text="Browse folder", command=fct_folder_save)
        self.button_saving_folder.place(relx=0.05, rely=0.53)

        self.counter = 0
        self.processing_thread = None

        self.label_processing = Label(window, text="",  fg="black")
        self.label_processing.place(relx=0.35, rely=0.7)

        self.button_start_processing = ttk.Button(window, text="Start process",
                                                  command=self.__start_processing)
        self.button_start_processing.place(relx=0.05, rely=0.75)
        self.stop_thread = False
        self.__shown_text = ""
        self.__num_of_processing_images = 0
        self.__update()
        
        ###################################################################################
        
        self.window.iconbitmap(os.path.join("Resources", 'elo-hyp_logo.ico'))

    def select_folder(self, storing_label):
        filename = filedialog.askdirectory(title="Select a Folder")
        storing_label.configure(text=filename)
        
    def select_file(self, storing_label):
        filename = filedialog.askopenfilename(title="Select a File")
        storing_label.configure(text=filename)

    def __update(self):
        if self.processing_thread is None:
            self.label_processing.configure(text=self.__shown_text)
        elif self.processing_thread.is_alive():
            self.label_processing.configure(text=f"Processing: {self.counter}/{self.__num_of_processing_images}.")
        else:
            self.label_processing.configure(text=self.__shown_text)

        self.window.after(1000, self.__update)

    def __process(self, input_dir: str, output_dir: str):
#     def __process(self, output_dir: str):
        try:
                    
################ load weights  ###################
            ssvm2 = ssvm.SSVM()
            ssvm2.load('post_process_data/PaviaU_red_osp.json')
                    
############## load data #####################
            data = scipy.io.loadmat(input_dir)['hyp_img_red']
            labels = sio.loadmat('data/PaviaU_gt.mat')['paviaU_gt']
            X = data.T
            y = np.array(labels).flatten()
            X = preprocessing.scale(X, axis=0)                    # Normalization
            I = [0, 1, 2,3,4,5,6,7,8,9]
            
                    # full data accuracy
            x_full_guess = ssvm2.predict(X)
            correct_prediction = x_full_guess - y
            N = len(y)
#             print('Total number of data for testing =', N)
            cp = len(correct_prediction[np.where(correct_prediction == 0)])
            Accuracy = len(correct_prediction[np.where(correct_prediction == 0)])*100/N
            print('Accuracy (in %) =', Accuracy)
            guess = []
            for i in range(10):
                L = N - sum(y == I[i]) + len(np.intersect1d(np.where(x_full_guess == I[i]),np.where(y == I[i])))
                guess = np.append(guess, L*100/N)
                    
####################################### to save the output  ###########################
            ### .txt file
            path_to_save = os.path.join(output_dir, "output_details_PaviaU_osp.txt")
            file1 = open(path_to_save, "w")
            file1.write("Overall Accuracy----\n")
            toFile = "Total number of data for testing ="
            toFile1 = "Correctly predicted = "
            toFile2 = "Overall Accuracy (in %) = "
            file1.write('\n{a} {b} \n{c} {d} \n{e} {f}\n'.format(a=toFile, b=N, c=toFile1, d=cp, e=toFile2, f=Accuracy))
            file1.write("\n----Class-wise Accuracy and Average Accuracy----\n")
            for i in range(10):
                F = "Accuracy of class %s" %I[i]
                file1.write('{a} (in %) = {b}\n'.format(a=F, b=guess[i]))
            File = "Average Accuracy (in %) = "
            file1.write( '{a} {b}'.format(a=File, b= np.mean(guess)))
            file1.close()
            ### figure
            plt.close("all")
            for i in range(10):
                plt.plot(ssvm2._w()[I[i]],label='Class %s' %I[i])
    
            lgd = plt.legend(title='sparsity', loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xlabel("Number of bands")
            plt.ylabel("Weights")
            plt.savefig(output_dir+'/SVM_weights_PaviaU_osp.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

            self.__shown_text = f"Results saved at: {output_dir}\n Overall Accuracy (in %) = {Accuracy}"
            self.button_start_processing.config(state='normal')
            self.counter = 0
        except Exception as ex:
            print(ex)
            self.__shown_text = "An exception occurred!"

    def __start_processing(self):
        
        input_dir = self.label_folder_imgs_path["text"]
        if input_dir is None or input_dir == "":
            messagebox.showerror("Error", "The processing folder must be selected!")
            return

        save_dir = self.label_saving_folder_path["text"]
        if save_dir is None or save_dir == "":
            messagebox.showerror("Error", "The saving folder must be selected!")
            return

        self.button_start_processing.config(state='disabled')

        self.processing_thread = threading.Thread(target=self.__process,
                                                  args=(input_dir, save_dir, ), daemon=True)
        self.processing_thread.start()
        
        
class ClassificationPaviaU_red_lpp:
    def __init__(self, window, window_title):

        self.window = window
        self.window.grab_set()  # Make it the main window.

        self.review_enroll_window = None
        self.settings_window_created = False

        self.window.title(window_title)
        self.window.minsize(750, 150)
        self.window.resizable(False, False)
        
        ####################################################################################

        # Folder input folder.
        self.label_folder_imgs = Label(window, text="Processing data:", fg="black", font='Arial 12')
        self.label_folder_imgs.place(relx=0.05, rely=0.05)
        self.label_folder_imgs_path = Label(window, text="", fg="black")
        self.label_folder_imgs_path.place(relx=0.35, rely=0.06)
        fct_folder_imgs = partial(self.select_file, self.label_folder_imgs_path)
        self.button_folder = ttk.Button(window, text="Browse file", command=fct_folder_imgs)
        self.button_folder.place(relx=0.05, rely=0.2)

        
        # Folder output folder.
        self.label_saving_folder = Label(window, text="Saving folder:", font='Arial 12')
        self.label_saving_folder.place(relx=0.05, rely=0.38)
        self.label_saving_folder_path = Label(window, text="")
        self.label_saving_folder_path.place(relx=0.35, rely=0.38)
        fct_folder_save = partial(self.select_folder, self.label_saving_folder_path)
        self.button_saving_folder = ttk.Button(window, text="Browse folder", command=fct_folder_save)
        self.button_saving_folder.place(relx=0.05, rely=0.53)

        self.counter = 0
        self.processing_thread = None

        self.label_processing = Label(window, text="",  fg="black")
        self.label_processing.place(relx=0.35, rely=0.7)

        self.button_start_processing = ttk.Button(window, text="Start process",
                                                  command=self.__start_processing)
        self.button_start_processing.place(relx=0.05, rely=0.75)
        self.stop_thread = False
        self.__shown_text = ""
        self.__num_of_processing_images = 0
        self.__update()
        
        ###################################################################################
        
        self.window.iconbitmap(os.path.join("Resources", 'elo-hyp_logo.ico'))

    def select_folder(self, storing_label):
        filename = filedialog.askdirectory(title="Select a Folder")
        storing_label.configure(text=filename)
        
    def select_file(self, storing_label):
        filename = filedialog.askopenfilename(title="Select a File")
        storing_label.configure(text=filename)

    def __update(self):
        if self.processing_thread is None:
            self.label_processing.configure(text=self.__shown_text)
        elif self.processing_thread.is_alive():
            self.label_processing.configure(text=f"Processing: {self.counter}/{self.__num_of_processing_images}.")
        else:
            self.label_processing.configure(text=self.__shown_text)

        self.window.after(1000, self.__update)

    def __process(self, input_dir: str, output_dir: str):
#     def __process(self, output_dir: str):
        try:
                    
################ load weights  ###################
            ssvm2 = ssvm.SSVM()
            ssvm2.load('post_process_data/PaviaU_red_osp.json')
                    
############## load data #####################
            data = scipy.io.loadmat(input_dir)['hyp_img_red']
            labels = sio.loadmat('data/PaviaU_gt.mat')['paviaU_gt']
            X = data.T
            y = np.array(labels).flatten()
            X = preprocessing.scale(X, axis=0)                    # Normalization
            I = [0, 1, 2,3,4,5,6,7,8,9]
            
                    # full data accuracy
            x_full_guess = ssvm2.predict(X)
            correct_prediction = x_full_guess - y
            N = len(y)
#             print('Total number of data for testing =', N)
            cp = len(correct_prediction[np.where(correct_prediction == 0)])
            Accuracy = len(correct_prediction[np.where(correct_prediction == 0)])*100/N
            print('Accuracy (in %) =', Accuracy)
            guess = []
            for i in range(10):
                L = N - sum(y == I[i]) + len(np.intersect1d(np.where(x_full_guess == I[i]),np.where(y == I[i])))
                guess = np.append(guess, L*100/N)
                    
####################################### to save the output  ###########################
            ### .txt file
            path_to_save = os.path.join(output_dir, "output_details_PaviaU_lpp.txt")
            file1 = open(path_to_save, "w")
            file1.write("Overall Accuracy----\n")
            toFile = "Total number of data for testing ="
            toFile1 = "Correctly predicted = "
            toFile2 = "Overall Accuracy (in %) = "
            file1.write('\n{a} {b} \n{c} {d} \n{e} {f}\n'.format(a=toFile, b=N, c=toFile1, d=cp, e=toFile2, f=Accuracy))
            file1.write("\n----Class-wise Accuracy and Average Accuracy----\n")
            for i in range(10):
                F = "Accuracy of class %s" %I[i]
                file1.write('{a} (in %) = {b}\n'.format(a=F, b=guess[i]))
            File = "Average Accuracy (in %) = "
            file1.write( '{a} {b}'.format(a=File, b= np.mean(guess)))
            file1.close()
            ### figure
            plt.close("all")
            for i in range(10):
                plt.plot(ssvm2._w()[I[i]],label='Class %s' %I[i])
    
            lgd = plt.legend(title='sparsity', loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xlabel("Number of bands")
            plt.ylabel("Weights")
            plt.savefig(output_dir+'/SVM_weights_PaviaU_lpp.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

            self.__shown_text = f"Results saved at: {output_dir}\n Overall Accuracy (in %) = {Accuracy}"
            self.button_start_processing.config(state='normal')
            self.counter = 0
        except Exception as ex:
            print(ex)
            self.__shown_text = "An exception occurred!"

    def __start_processing(self):
        
        input_dir = self.label_folder_imgs_path["text"]
        if input_dir is None or input_dir == "":
            messagebox.showerror("Error", "The processing folder must be selected!")
            return

        save_dir = self.label_saving_folder_path["text"]
        if save_dir is None or save_dir == "":
            messagebox.showerror("Error", "The saving folder must be selected!")
            return

        self.button_start_processing.config(state='disabled')

        self.processing_thread = threading.Thread(target=self.__process,
                                                  args=(input_dir, save_dir, ), daemon=True)
        self.processing_thread.start()
