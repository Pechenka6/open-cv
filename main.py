import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageOps
import cv2
import numpy as np
import ctypes

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Application")
        self.image = None 
        self.processed_image = None  
        self.setup_ui()

    def setup_ui(self):
        ttk.Button(self.root, text="Load Image", command=self.load_image).grid(row=0, column=0, padx=10, pady=10)
        ttk.Button(self.root, text="Save Image", command=self.save_image).grid(row=0, column=1, padx=10, pady=10)
        ttk.Button(self.root, text="Equalize RGB Histogram", command=self.equalize_histogram_rgb).grid(row=0, column=2, padx=10, pady=10)
        ttk.Button(self.root, text="Equalize HSV Histogram", command=self.equalize_histogram_hsv).grid(row=0, column=3, padx=10, pady=10)
        ttk.Button(self.root, text="Edge Detection", command=self.detect_edges).grid(row=0, column=4, padx=10, pady=10)
        ttk.Button(self.root, text="Adaptive Threshold", command=self.adaptive_threshold).grid(row=0, column=5, padx=10, pady=10)
        ttk.Button(self.root, text="Morphological Transform", command=self.morphological_transform).grid(row=0, column=6, padx=10, pady=10)

        self.threshold_slider = ttk.Scale(self.root, from_=0, to=255, orient="horizontal", command=self.apply_threshold)
        self.threshold_slider.set(128)
        self.threshold_slider.grid(row=0, column=7, padx=10, pady=10)

        self.original_canvas = tk.Canvas(self.root, width=800, height=600, bg="gray")
        self.original_canvas.grid(row=1, column=0, columnspan=4, padx=10, pady=10)

        self.processed_canvas = tk.Canvas(self.root, width=800, height=600, bg="gray")
        self.processed_canvas.grid(row=1, column=4, columnspan=4, padx=10, pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            self.image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
            self.display_image(self.image, self.original_canvas)

    def save_image(self):
        if self.processed_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if file_path:
                cv2.imwrite(file_path, cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR))

    def display_image(self, image, canvas):
        image = Image.fromarray(image)
        image = ImageOps.fit(image, (800, 600), Image.Resampling.LANCZOS)
        tk_image = ImageTk.PhotoImage(image)

        canvas.image = tk_image
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)

    def equalize_histogram_rgb(self):
        if self.image is not None:
            channels = cv2.split(self.image)
            eq_channels = [cv2.equalizeHist(channel) for channel in channels]
            self.processed_image = cv2.merge(eq_channels)
            self.display_image(self.processed_image, self.processed_canvas)

    def equalize_histogram_hsv(self):
        if self.image is not None:
            hsv_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
            hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])
            self.processed_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
            self.display_image(self.processed_image, self.processed_canvas)

    def detect_edges(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray_image, 100, 200)
            self.processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            self.display_image(self.processed_image, self.processed_canvas)

    def apply_threshold(self, value):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
            _, thresholded = cv2.threshold(gray_image, int(float(value)), 255, cv2.THRESH_BINARY)
            self.processed_image = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
            self.display_image(self.processed_image, self.processed_canvas)

    def adaptive_threshold(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
            adaptive = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            self.processed_image = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2RGB)
            self.display_image(self.processed_image, self.processed_canvas)

    def morphological_transform(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            morph = cv2.morphologyEx(gray_image, cv2.MORPH_GRADIENT, kernel)
            self.processed_image = cv2.cvtColor(morph, cv2.COLOR_GRAY2RGB)
            self.display_image(self.processed_image, self.processed_canvas)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
