'''
Ghailan Fadah
2/2/26
CS5330
lightweight tinker GUI for CBIR project
'''

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import subprocess
import os
import re

CBIR_EXEC = "./run_CBIR"        # path to compiled C++ binary
IMAGE_DIR = "./data"     # dataset directory

class CBIRGui:
    def __init__(self, root):
        self.root = root
        self.root.title("CBIR")

        self.target_image_path = None
        self.image_refs = [] 

        self.build_ui()

    def build_ui(self):
        
        self.target_label = tk.Label(self.root, text="Target Image")
        self.target_label.pack(pady=5)

        self.target_img_panel = tk.Label(self.root)
        self.target_img_panel.pack()

        tk.Button(self.root, text="Select Image", command=self.select_image).pack(pady=5)

        
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)

        tk.Label(control_frame, text="Top N:").grid(row=0, column=0)
        self.top_n = tk.Entry(control_frame, width=5)
        self.top_n.insert(0, "5")
        self.top_n.grid(row=0, column=1)

        tk.Label(control_frame, text="Feature:").grid(row=0, column=2)
        self.feature_var = tk.StringVar(value="baseline")
        ttk.Combobox(
            control_frame,
            textvariable=self.feature_var,
            values=["baseline", "hist", "2hist", "hist-text", "sun", "dnn"],
            width=10
        ).grid(row=0, column=3)

        tk.Label(control_frame, text="Matching:").grid(row=0, column=4)
        self.match_var = tk.StringVar(value="SSD")
        ttk.Combobox(
            control_frame,
            textvariable=self.match_var,
            values=["SSD", "i"],
            width=8
        ).grid(row=0, column=5)

        tk.Button(self.root, text="Run CBIR", command=self.run_cbir).pack(pady=10)

        
        self.canvas = tk.Canvas(self.root, height=400)
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.results_frame = tk.Frame(self.canvas)

        self.results_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.results_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

    
    def select_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.png *.ppm *.tif")]
        )
        if not path:
            return

        self.target_image_path = path
        img = Image.open(path).resize((250, 150))
        img_tk = ImageTk.PhotoImage(img)
        self.target_img_panel.configure(image=img_tk)
        self.target_img_panel.image = img_tk

    # ===== Run CBIR =====
    def run_cbir(self):
        if not self.target_image_path:
            messagebox.showerror("Error", "Please select a target image.")
            return

        # clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        self.image_refs.clear()

        cmd = [
            CBIR_EXEC,
            self.target_image_path,
            IMAGE_DIR,
            self.feature_var.get(),
            self.match_var.get(),
            self.top_n.get()
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            messagebox.showerror("CBIR Error", e.stderr)
            return

        self.parse_and_display(result.stdout)

    # ===== Parse output =====
    def parse_and_display(self, output):

        for line in output.splitlines():
            path, dist = line.split(",")
            self.add_result(path, dist)

    
       

    
    def add_result(self, img_path, distance):
        frame = tk.Frame(self.results_frame, pady=5)
        frame.pack(fill="x")
        img_path = img_path.strip()
        img = Image.open(img_path).resize((120, 80))
        img_tk = ImageTk.PhotoImage(img)
        self.image_refs.append(img_tk)

        img_label = tk.Label(frame, image=img_tk)
        img_label.pack(side="left", padx=5)

        text = f"{os.path.basename(img_path)}\nDistance: {distance}"
        tk.Label(frame, text=text, justify="left").pack(side="left")

# ===== Run app =====
if __name__ == "__main__":
    root = tk.Tk()
    app = CBIRGui(root)
    root.mainloop()

