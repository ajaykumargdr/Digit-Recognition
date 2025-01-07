# Handwriting Recognition ✍🏻💻

This project features a handwriting recognition system powered by a **Convolutional Neural Network (CNN)**, built using **TensorFlow** and **Keras**. It is designed to accurately recognize handwritten digits, leveraging data from the **MNIST dataset**.

## 📊 Model performance

### Training and validation accuracy
The graph below shows the training and validation accuracy of the model :

![Figure_1](https://github.com/user-attachments/assets/f6eee4da-f2a0-467d-a370-93a73fe15925)

## ⚙️ Features 

- Train a CNN model on the MNIST dataset.
- Save and load the trained model.
- GUI for loading images and drawing digits on a canvas.
- Real-time prediction of handwritten digits with confidence scores.
- Visualization of training history and predictions.

   <img src="https://github.com/user-attachments/assets/f789701f-52ab-4e27-9621-ee2a858d3636" width="250" />

## 🛠️ Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/soroqn1/Digit-Recognition
    cd Digit-Recognition
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## 🔍 Usage

### Training the Model

1. Run the `general.py` script to train the model:
    ```sh
    python general.py
    ```

2. The trained model will be saved as `models/handwriting_recognition_model.h5`.

### Running the GUI

1. Run the `gui.py` script to start the GUI application:
    ```sh
    python gui.py
    ```

2. Use the GUI to load an image or draw a digit on the canvas for prediction.

## 🗂️ Files

- `general.py`: Script for training the CNN model on the MNIST dataset.
- `gui.py`: Script for the GUI application to load images and draw digits for prediction.
- `requirements.txt`: List of required Python packages.


## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The MNIST dataset is provided by Yann LeCun and Corinna Cortes.
- TensorFlow and Keras are open-source libraries developed by the TensorFlow team.
```
