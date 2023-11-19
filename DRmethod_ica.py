from tkinter import  Label, filedialog, messagebox
from tkinter import ttk
import threading
#import time

from scipy.io import loadmat, savemat
import dr_methods.fastICA as ica
from functools import partial
import os




class DRWindow_ica:
    def __init__(self, window, window_title):

        self.window = window
        self.window.grab_set()  # Make it the main window.

        self.review_enroll_window = None
        self.settings_window_created = False

        self.window.title(window_title)
        self.window.minsize(600, 150)
        self.window.resizable(False, False)
        
        ####################################################################################
            
        self.label_load_imgs = Label(window, text="Hyp-Image:")
        self.label_load_imgs.place(relx=0.05, rely=0.19)
        self.label_saving_file_path = Label(window, text="", fg="black")
        self.label_saving_file_path.place(relx=0.2, rely=0.2)
        
        fct_hyp_img_load = partial(self.openfile, self.label_saving_file_path)
        
        self.button_load_hypImg = ttk.Button(window, 
                    text="Load Hyp-Image", command= fct_hyp_img_load).place(relx=0.05, rely=0.05)
        
          
        # Folder output folder.
        self.label_saving_folder = Label(window, text="Saving folder:")
        self.label_saving_folder.place(relx=0.05, rely=0.38)
        
        self.label_saving_folder_path = Label(window, text="")
        self.label_saving_folder_path.place(relx=0.2, rely=0.38)
        
        fct_folder_save = partial(self.select_folder, self.label_saving_folder_path)
        self.button_saving_folder = ttk.Button(window, text="Browse folder",
                                               command=fct_folder_save)
        self.button_saving_folder.place(relx=0.05, rely=0.53)
        
        
        # Start Process 
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
          foldername = filedialog.askdirectory(title="Select a Folder")
          storing_label.configure(text=foldername)

    def openfile(self, hyp_filename):
           #global hyp_file_path
           
           filename =filedialog.askopenfilename(title="Select a .mat file")
           hyp_filename.configure(text=filename)
         
        
    #########################################################################################

    def __update(self):
        
        if self.processing_thread is None:
            self.label_processing.configure(text=self.__shown_text)
        elif self.processing_thread.is_alive():
            #self.label_processing.configure(text=f"Processing: {self.counter}/{self.__num_of_processing_images}.")
            self.label_processing.configure(text="Processing")
        else:
            self.label_processing.configure(text=self.__shown_text)

        self.window.after(1000, self.__update)

    def __process(self, output_dir: str):
        try:
            hyp_file_path = self.label_saving_file_path["text"]
            hyp_dic = loadmat(hyp_file_path) # give the name of the file
            hyp_name =  list(hyp_dic)[-1] # get the name of the variable in the dict
            hyp_img = hyp_dic[hyp_name] # get the value
            
            if hyp_img.ndim == 3:
                hyp_img = hyp_img.reshape(-1, hyp_img.shape[2])
                
            if hyp_img.shape[0]< hyp_img.shape[1]:
                hyp_img = hyp_img.T

            no_bands = 10
            dr_ica = ica.FastICA(n_bands = no_bands)

            dr_ica.fit(hyp_img)
            hyp_img_red = dr_ica.transform(hyp_img)
            
            print(hyp_img_red.shape)

            #hyp_img_recover = dr_pca.inverse_transform(hyp_img_red)
            
            ############### save output  ########################
           # path_to_save_projection = os.path.join(output_dir, "projection_ica")
            path_to_save_imgred = os.path.join(output_dir, "reduced_hypImg_ica.mat")
            
            savemat(path_to_save_imgred, {'hyp_img_red' : hyp_img_red})
            #dr_ica.save(path_to_save_projection)
            self.button_start_processing.config(state='normal')
            self.counter = 0
        except Exception as ex:
            print(ex)
            self.__shown_text = "An exception occurred!"

    def __start_processing(self):
        
        load_img = self.label_saving_file_path["text"]
        if load_img is None or load_img == "":
            messagebox.showerror("Error", "A hyperspectral image must be selected!")
            return

        save_dir = self.label_saving_folder_path["text"]
        if save_dir is None or save_dir == "":
            messagebox.showerror("Error", "The saving folder must be selected!")
            return

        self.button_start_processing.config(state='disabled')

        self.processing_thread = threading.Thread(target=self.__process,
                                                  args=(save_dir, ), daemon=True)
        self.processing_thread.start()
