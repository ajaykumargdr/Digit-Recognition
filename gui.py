import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Значения mean и std из этапа обучения модели
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

        self.result_label = tk.Label(root, text="", font=("Arial", 14), fg="blue")
        self.result_label.pack()

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp"), ("All Files", "*.*")]
        )
        if file_path:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, file_path)

    def predict_digit(self):
        image_path = self.path_entry.get()
        if not image_path:
            messagebox.showerror("Error", "Please enter a valid image path!")
            return

        try:
            # Load and process the image
            img = Image.open(image_path)
            img = img.convert('L')  # Convert to grayscale
            img = img.resize((28, 28), Image.LANCZOS)  # Resize to 28x28 pixels

            # Normalize the image
            img_array = np.array(img).astype('float32') / 255.0
            img_array = (img_array - MEAN) / STD
            img_array = img_array.reshape(1, 28, 28, 1)

            # Predict
            prediction = self.model.predict(img_array)
            digit = np.argmax(prediction[0])
            confidence = float(prediction[0][digit])

            # Show result
            self.result_label.config(
                text=f"Predicted Digit: {digit}\nConfidence: {confidence:.2%}"
            )
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Path Image Recognition")
    app = PathRecognitionApp(root)
    root.mainloop()
