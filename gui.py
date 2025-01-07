import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageGrab, ImageOps, ImageDraw
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import io
import platform

# Mean and std values from the model training phase
MEAN = 0.1307
STD = 0.3081

class PathRecognitionApp:
    def __init__(self, root):
        self.root = root

        # Load the model
        try:
            self.model = load_model('models/handwriting_recognition_model.h5')
        except Exception as e:
            messagebox.showerror("Error", f"Could not load the model: {e}")
            self.root.destroy()
            return

        # Configure GUI
        self.path_label = tk.Label(root, text="Enter image path:", font=("Arial", 12))
        self.path_label.pack()

        self.path_entry = tk.Entry(root, width=50, font=("Arial", 12))
        self.path_entry.pack()

        self.browse_btn = tk.Button(root, text="Browse", command=self.browse_image)
        self.browse_btn.pack()

        self.predict_btn = tk.Button(root, text="Predict", command=self.predict_digit)
        self.predict_btn.pack()

        self.result_label = tk.Label(root, text="", font=("Arial", 14), fg="white")
        self.result_label.pack()

        self.setup_canvas()

    def setup_canvas(self):
        # Create a frame to hold the canvas
        self.canvas_frame = tk.Frame(self.root, padx=10, pady=10)
        self.canvas_frame.pack()
        
        # Create canvas with white background
        self.canvas = tk.Canvas(
            self.canvas_frame,
            width=280,
            height=280,
            bg="white",
            highlightthickness=1,
            highlightbackground="black"
        )
        self.canvas.pack()

        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)
        self.canvas.bind("<Button-1>", self.set_last_coords)
        self.canvas.bind("<ButtonRelease-1>", lambda e: self.canvas.update())

        # Create buttons
        self.clear_btn = tk.Button(self.root, text="Clear Canvas", command=self.clear_canvas)
        self.clear_btn.pack()

        self.predict_canvas_btn = tk.Button(self.root, text="Predict from Canvas", command=self.predict_from_canvas)
        self.predict_canvas_btn.pack()

    def set_last_coords(self, event):
        self.lastX, self.lastY = event.x, event.y

    def browse_image(self):
        try:
            # Special handling for macOS
            if platform.system() == 'Darwin':
                root = tk.Tk()
                root.withdraw()  # Hide the main window
                root.call('wm', 'attributes', '.', '-topmost', True)  # Bring dialog to the front
                
            file_path = filedialog.askopenfilename(
                parent=self.root,
                title="Select an image",
                filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp"), ("All Files", "*.*")]
            )
            
            if file_path:
                self.path_entry.delete(0, tk.END)
                self.path_entry.insert(0, file_path)
            
            # Close the temporary window for macOS
            if platform.system() == 'Darwin':
                root.destroy()
                
        except Exception as e:
            messagebox.showerror("Error", f"Error selecting file: {str(e)}")

    def process_image(self, img):
        # Convert to numpy array if PIL Image
        if isinstance(img, Image.Image):
            img = np.array(img)

        # Ensure image is grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Invert colors (black on white to white on black)
        img = 255 - img

        # Add padding to make the image square
        height, width = img.shape
        if height > width:
            diff = height - width
            padding = diff // 2
            img = np.pad(img, ((0, 0), (padding, diff - padding)), 'constant', constant_values=0)
        else:
            diff = width - height
            padding = diff // 2
            img = np.pad(img, ((padding, diff - padding), (0, 0)), 'constant', constant_values=0)

        # Find bounding box of digit
        coords = cv2.findNonZero(img)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            # Add padding around the digit
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2*padding)
            h = min(img.shape[0] - y, h + 2*padding)
            img = img[y:y+h, x:x+w]

        # Resize to 28x28
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

        # Normalize
        img = img.astype('float32') / 255.0
        img = (img - MEAN) / STD

        return img

    def process_canvas_image(self, img):
        # Convert PIL to numpy array
        img_array = np.array(img)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        img_array = cv2.GaussianBlur(img_array, (5, 5), 0)
        
        # Apply threshold to make the digit more distinct
        _, img_array = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)
        
        # Invert the image (white on black)
        img_array = 255 - img_array
        
        # Find the bounding box of the digit
        coords = cv2.findNonZero(img_array)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            # Add padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img_array.shape[1] - x, w + 2*padding)
            h = min(img_array.shape[0] - y, h + 2*padding)
            img_array = img_array[y:y+h, x:x+w]
            
            # Make it square
            size = max(w, h)
            square = np.zeros((size, size), dtype=np.uint8)
            square[:h, :w] = img_array
            
            # Resize to 28x28
            img_array = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
            
            # Normalize
            img_array = img_array.astype('float32') / 255.0
            img_array = (img_array - MEAN) / STD

        return img_array

    def predict_digit(self):
        image_path = self.path_entry.get()
        if not image_path:
            messagebox.showerror("Error", "Please enter a valid image path!")
            return

        try:
            img = Image.open(image_path)
            img = img.convert('L')
            img_processed = self.process_image(img)
            
            # Make prediction
            img_array = img_processed.reshape(1, 28, 28, 1)
            prediction = self.model.predict(img_array)
            digit = np.argmax(prediction[0])
            confidence = float(prediction[0][digit])

            self.result_label.config(
                text=f"Predicted Digit: {digit}\nConfidence: {confidence:.2%}"
            )

            # Debug visualization
            self.show_debug_image(img_processed)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def draw_on_canvas(self, event):
        x, y = event.x, event.y
        r = 8  # radius of each point
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black', outline='black')
        if hasattr(self, 'lastX') and hasattr(self, 'lastY'):
            self.canvas.create_line(
                self.lastX, self.lastY, x, y,
                width=r*2,
                fill="black",
                capstyle=tk.ROUND,
                smooth=True
            )
        self.lastX, self.lastY = x, y

    def clear_canvas(self):
        self.canvas.delete("all")

    def predict_from_canvas(self):
        try:
            # Get canvas dimensions
            width = self.canvas.winfo_width()
            height = self.canvas.winfo_height()

            # Create a new image with white background
            image = Image.new('L', (width, height), color='white')
            draw = ImageDraw.Draw(image)

            # Get all canvas objects
            for item in self.canvas.find_all():
                # Get object coordinates
                coords = self.canvas.coords(item)
                if len(coords) >= 4:  # Line or oval
                    # Draw with thick black line
                    draw.line(coords, fill='black', width=16)

            # Convert to array and process
            img_array = np.array(image)
            
            # Apply threshold
            _, img_array = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)
            
            # Process the image using existing process_image method
            img_processed = self.process_image(img_array)
            
            # Make prediction
            img_array = img_processed.reshape(1, 28, 28, 1)
            prediction = self.model.predict(img_array)
            digit = np.argmax(prediction[0])
            confidence = float(prediction[0][digit])
            
            self.result_label.config(
                text=f"Predicted Digit: {digit}\nConfidence: {confidence:.2%}",
                fg="white"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Path Image Recognition")
    app = PathRecognitionApp(root)
    root.mainloop()
