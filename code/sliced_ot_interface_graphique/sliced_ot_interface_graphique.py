import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
#from ttkthemes import ThemedTk, ThemedStyle

class TransportOptimal:
    def __init__(self, src, trgt, nb_steps):
        self.src = src
        self.trgt = trgt
        self.steps = nb_steps
    
    def draw_random_direction(self, d):
        direction = np.random.normal(0, 1, d)
        return direction / np.linalg.norm(direction)

    def sort_projection(self, samples, direction, h, w):
        projections = (samples @ direction).reshape((h * w))
        sorted_indices = np.argsort(projections)
        return sorted_indices
    
    def sliced_ot_color_transfer_no_animation(self):
        h, w, c = self.src.shape
        source_dtype = self.src.dtype

        if (h, w, c) != self.trgt.shape:
            raise ValueError("source and target shapes must be equal")

        new_source = self.src.copy()
        for i in range(self.steps):
            advect = np.zeros((h * w, c), dtype=source_dtype)

            direction = self.draw_random_direction(c)
            idSource = self.sort_projection(new_source, direction, h, w)
            idTarget = self.sort_projection(self.trgt, direction, h, w)

            projsource = (new_source @ direction).reshape((h * w))
            projtarget = (self.trgt @ direction).reshape((h * w))

            displacements = projtarget[idTarget] - projsource[idSource]
            for i_c in range(c):
                advect[idSource, i_c] += displacements * direction[i_c]

            new_source += advect.reshape((h, w, c))

        new_source = np.clip(new_source, 0, 1).astype(float)

        return new_source
    
    def sliced_ot_color_transfer_animation(self):
        h, w, c = self.src.shape
        source_dtype = self.src.dtype
        new_source = self.src.copy()
        images,hists = [],[]
        for i in range(self.steps):
            advect = np.zeros((h * w, c), dtype=source_dtype)

            direction = self.draw_random_direction(c)
            idSource = self.sort_projection(new_source, direction, h, w)
            idTarget = self.sort_projection(self.trgt, direction, h, w)

            projsource = (new_source @ direction).reshape((h * w))
            projtarget = (self.trgt @ direction).reshape((h * w))

            displacements = projtarget[idTarget] - projsource[idSource]
            for i_c in range(c):
                advect[idSource, i_c] += displacements * direction[i_c]

            new_source += advect.reshape((h, w, c))
            histogram_r, histogram_g, histogram_b = InterfaceGraphique.calcul_histo(new_source)

            images.append(new_source.astype(float))
            hists.append([histogram_r,histogram_g,histogram_b])

        return images,hists
    
    def sliced_ot_color_transfer_batches_regularisation(self, batch_size):
        h, w, c = self.src.shape
        source_dtype = self.src.dtype
        new_source = self.src.copy()
        images,hists = [],[]
        batch_idx = 0
        for i in range(self.steps):
            advect = np.zeros((h * w, c), dtype=source_dtype)

            direction = self.draw_random_direction(c)
            idSource = self.sort_projection(new_source, direction, h, w)
            idTarget = self.sort_projection(self.trgt, direction, h, w)

            projsource = (new_source @ direction).reshape((h * w))
            projtarget = (self.trgt @ direction).reshape((h * w))

            displacements = projtarget[idTarget] - projsource[idSource]
            for i_c in range(c):
                advect[idSource, i_c] += displacements * direction[i_c]

            batch_idx += 1
            if batch_idx == batch_size:
                new_source += advect.reshape((h, w, c)) 
                batch_idx = 0

            histogram_r, histogram_g, histogram_b = InterfaceGraphique.calcul_histo(new_source)

            images.append(new_source.astype(float))
            hists.append([histogram_r,histogram_g,histogram_b])

        return images,hists

    def sliced_ot_color_transfer_interpolation(self, batch_size, alpha):
        h, w, c = self.src.shape
        source_dtype = self.src.dtype
        new_source = self.src.copy()
        images, hists = [], []
        batch_idx = 0
        for i in range(self.steps):
            advect = np.zeros((h * w, c), dtype=source_dtype)

            direction = self.draw_random_direction(c)
            idSource = self.sort_projection(new_source, direction, h, w)
            idTarget = self.sort_projection(self.trgt, direction, h, w)

            projsource = (new_source @ direction).reshape((h * w))
            projtarget = (self.trgt @ direction).reshape((h * w))

            displacements = alpha * (projtarget[idTarget] - projsource[idSource])
            for i_c in range(c):
                advect[idSource, i_c] += displacements * direction[i_c]

            batch_idx += 1
            if batch_idx == batch_size:
                new_source += advect.reshape((h, w, c))
                batch_idx = 0
            histogram_r, histogram_g, histogram_b = InterfaceGraphique.calcul_histo(new_source)
            images.append(new_source.astype(float))
            hists.append([histogram_r, histogram_g, histogram_b])

        return images, hists
    
class InterfaceGraphique:
    def __init__(self, master):
        self.master = master
        master.title("TRANSPORT OPTIMAL")

        style = ttk.Style()
        style.configure("TButton", padding=10, relief="flat", background="black", foreground="black")

        self.label = tk.Label(master, text="Choisir le mode", font=("Arial", 12))
        self.label.pack(pady=10)

        self.create_button_with_entry("Transport optimal sans animation")
        self.create_button_with_entry("Transport optimal avec animation")
        self.create_button_with_entry("Transport optimal batches et regularisation avec animation")
        self.create_button_with_entry("Transport optimal interpolation avec animation")

        self.bouton_quitter = ttk.Button(master, text="Quitter", command=master.quit)
        self.bouton_quitter.pack(pady=10)

    def create_button_with_entry(self, button_text):
        frame = ttk.Frame(self.master)
        frame.pack(pady=5, padx=10, fill="both")

        bouton = ttk.Button(frame, text=button_text, command=lambda: self.open_sub_interface(button_text))
        bouton.pack(side="left")

    def open_sub_interface(self, button_text):
        sub_window = tk.Toplevel(self.master)
        sub_window.geometry("700x500")
        sub_window.title(f"Options pour {button_text}")

        label = tk.Label(sub_window, text=f"Options pour {button_text}", font=("Arial", 12))
        label.pack(pady=10)

        # Champ de saisie pour le nombre de steps
        nb_steps_label = tk.Label(sub_window, text="Nombre de steps:")
        nb_steps_label.pack()

        nb_steps_entry = ttk.Entry(sub_window)
        nb_steps_entry.pack(pady=5)

        nb_batches_entry = None
        alpha_entry = None
        if button_text != "Transport optimal sans animation" and button_text != "Transport optimal avec animation":
            nb_batches_label = tk.Label(sub_window, text="Batches:")
            nb_batches_label.pack()

            nb_batches_entry = ttk.Entry(sub_window)
            nb_batches_entry.pack(pady=5)
            if button_text == "Transport optimal interpolation avec animation":
                alpha_label = tk.Label(sub_window, text="Alpha:")
                alpha_label.pack()

                alpha_entry = ttk.Entry(sub_window)
                alpha_entry.pack(pady=5)

        # Variable pour stocker le chemin de l'image sélectionnée
        image_path_src = tk.StringVar()

        image_path_trgt = tk.StringVar()

        # Bouton de sélection d'image
        bouton_image_src = ttk.Button(sub_window, text="Sélectionner Image Source", command=lambda: self.select_image(sub_window, button_text, image_path_src, bouton_image_src))
        bouton_image_src.pack(pady=10)

        bouton_image_trgt = ttk.Button(sub_window, text="Sélectionner Image Cible", command=lambda: self.select_image(sub_window, button_text, image_path_trgt, bouton_image_trgt))
        bouton_image_trgt.pack(pady=10)

        # Bouton Afficher
        bouton_afficher = ttk.Button(sub_window, text="Afficher", command=lambda: self.afficher(nb_steps_entry.get(), nb_batches_entry, alpha_entry, image_path_src.get(), image_path_trgt.get(), button_text))
        bouton_afficher.pack(pady=10)

    def calcul_histo(img):
        histogram_r = np.histogram(img.reshape(-1, 3)[:,0], bins=256)
        histogram_g = np.histogram(img.reshape(-1, 3)[:,1], bins=256)
        histogram_b = np.histogram(img.reshape(-1, 3)[:,2], bins=256)
        return histogram_r, histogram_g, histogram_b
        
    def update(self, i, imgs, hists, ax):
        ax[2][0].clear()
        ax[2][1].clear()
        ax[2][0].imshow(imgs[i])
        ax[2][1].plot(range(len(hists[i][0][0])), hists[i][0][0], color='r')
        ax[2][1].plot(range(len(hists[i][1][0])), hists[i][1][0], color='g')
        ax[2][1].plot(range(len(hists[i][2][0])), hists[i][2][0], color='b')

    def select_image(self, sub_window, button_text, image_path, bouton_image):
        file_path = filedialog.askopenfilename(title="Sélectionner une image", filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
        if file_path:
            print(f"Image sélectionnée pour {button_text}: {file_path}")
            image_path.set(file_path)
            bouton_image["text"] = "Image Sélectionnée"
            sub_window.after(20000000, lambda: bouton_image.config(text="Sélectionner Image"))  # Réinitialise le texte après 20000000 ms
        else:
            bouton_image["text"] = "Sélectionner Image"  # Réinitialise le texte si aucune image n'est sélectionnée

    def afficher(self, nb_steps, nb_batches, alpha, image_path_src, image_path_trgt, button_text):
        src, trgt = plt.imread(image_path_src),plt.imread(image_path_trgt)
        OT = TransportOptimal(src.astype(float), trgt.astype(float), int(nb_steps))
        if button_text == "Transport optimal sans animation":
            fig = plt.figure()
            res = OT.sliced_ot_color_transfer_no_animation()

            # Créer une nouvelle fenêtre pour les plots et animations
            plot_window = tk.Toplevel(self.master)
            plot_window.geometry("800x600")
            plot_window.title(f"Plots et Animations pour {button_text}")

            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.get_tk_widget().pack()

            toolbar = NavigationToolbar2Tk(canvas, plot_window)
            canvas.get_tk_widget().pack()

            # Afficher la figure dans la fenêtre
            canvas.draw()
            plt.imshow(res)
        elif button_text == "Transport optimal avec animation":
            images,hists = OT.sliced_ot_color_transfer_animation()
            fig, ax = plt.subplots(3, 2, figsize=(10, 4)) # Create subplots for image and histogram
            ax[0][0].imshow(src)
            histogram_r, histogram_g, histogram_b = InterfaceGraphique.calcul_histo(src)
            ax[0][1].plot(range(len(histogram_r[0])), histogram_r[0], color='r')
            ax[0][1].plot(range(len(histogram_g[0])), histogram_g[0], color='g')
            ax[0][1].plot(range(len(histogram_b[0])), histogram_b[0], color='b')

            ax[1][0].imshow(trgt)

            histogram_r, histogram_g, histogram_b = InterfaceGraphique.calcul_histo(trgt)
            ax[1][1].plot(range(len(histogram_r[0])), histogram_r[0], color='r')
            ax[1][1].plot(range(len(histogram_g[0])), histogram_g[0], color='g')
            ax[1][1].plot(range(len(histogram_b[0])), histogram_b[0], color='b')

            # Créer une nouvelle fenêtre pour les plots et animations
            plot_window = tk.Toplevel(self.master)
            plot_window.geometry("800x600")
            plot_window.title(f"Plots et Animations pour {button_text}")

            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.get_tk_widget().pack()

            toolbar = NavigationToolbar2Tk(canvas, plot_window)
            canvas.get_tk_widget().pack()

            # Afficher la figure dans la fenêtre
            ani = FuncAnimation(fig, lambda i : self.update(i=i, imgs = images, hists = hists, ax = ax), frames=len(images), interval=10, repeat=False)
            canvas.draw()

            plt.show()
        elif button_text == "Transport optimal batches et regularisation avec animation":
            images,hists = OT.sliced_ot_color_transfer_batches_regularisation(batch_size= int(nb_batches.get()))
            images[-1] = cv2.bilateralFilter(images[-1].astype(np.float32), 0, 0.5, 0.5)
            fig, ax = plt.subplots(3, 2, figsize=(10, 4)) # Create subplots for image and histogram
            ax[0][0].imshow(src)
            histogram_r, histogram_g, histogram_b = InterfaceGraphique.calcul_histo(src)
            ax[0][1].plot(range(len(histogram_r[0])), histogram_r[0], color='r')
            ax[0][1].plot(range(len(histogram_g[0])), histogram_g[0], color='g')
            ax[0][1].plot(range(len(histogram_b[0])), histogram_b[0], color='b')

            ax[1][0].imshow(trgt)

            histogram_r, histogram_g, histogram_b = InterfaceGraphique.calcul_histo(trgt)
            ax[1][1].plot(range(len(histogram_r[0])), histogram_r[0], color='r')
            ax[1][1].plot(range(len(histogram_g[0])), histogram_g[0], color='g')
            ax[1][1].plot(range(len(histogram_b[0])), histogram_b[0], color='b')

            # Créer une nouvelle fenêtre pour les plots et animations
            plot_window = tk.Toplevel(self.master)
            plot_window.geometry("800x600")
            plot_window.title(f"Plots et Animations pour {button_text}")

            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.get_tk_widget().pack()

            toolbar = NavigationToolbar2Tk(canvas, plot_window)
            canvas.get_tk_widget().pack()

            # Afficher la figure dans la fenêtre
            ani = FuncAnimation(fig, lambda i : self.update(i=i, imgs = images, hists = hists, ax = ax), frames=len(images), interval=10, repeat=False)
            canvas.draw()
            plt.show()
        elif button_text == "Transport optimal interpolation avec animation":
            images,hists = OT.sliced_ot_color_transfer_interpolation(batch_size= int(nb_batches.get()), alpha= float(alpha.get()))
            images[-1] = cv2.bilateralFilter(images[-1].astype(np.float32), 0, 0.5, 0.5)
            fig, ax = plt.subplots(3, 2, figsize=(10, 4)) # Create subplots for image and histogram
            ax[0][0].imshow(src)
            histogram_r, histogram_g, histogram_b = InterfaceGraphique.calcul_histo(src)
            ax[0][1].plot(range(len(histogram_r[0])), histogram_r[0], color='r')
            ax[0][1].plot(range(len(histogram_g[0])), histogram_g[0], color='g')
            ax[0][1].plot(range(len(histogram_b[0])), histogram_b[0], color='b')

            ax[1][0].imshow(trgt)

            histogram_r, histogram_g, histogram_b = InterfaceGraphique.calcul_histo(trgt)
            ax[1][1].plot(range(len(histogram_r[0])), histogram_r[0], color='r')
            ax[1][1].plot(range(len(histogram_g[0])), histogram_g[0], color='g')
            ax[1][1].plot(range(len(histogram_b[0])), histogram_b[0], color='b')

            # Créer une nouvelle fenêtre pour les plots et animations
            plot_window = tk.Toplevel(self.master)
            plot_window.geometry("800x600")
            plot_window.title(f"Plots et Animations pour {button_text}")

            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.get_tk_widget().pack()

            toolbar = NavigationToolbar2Tk(canvas, plot_window)
            canvas.get_tk_widget().pack()

            # Afficher la figure dans la fenêtre
            ani = FuncAnimation(fig, lambda i : self.update(i=i, imgs = images, hists = hists, ax = ax), frames=len(images), interval=10, repeat=False)
            canvas.draw()

            plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = InterfaceGraphique(root)
    root.geometry("400x400")  
    root.mainloop()