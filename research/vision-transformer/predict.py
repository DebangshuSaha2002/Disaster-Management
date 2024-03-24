import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from transformers import ViTFeatureExtractor, ViTForImageClassification
from hugsvision.inference.VisionClassifierInference import VisionClassifierInference

def browse_image():
    filename = filedialog.askopenfilename(initialdir="/", title="Select Image", filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")))
    if filename:
        predict_image(filename)

def predict_image(img_path):
    feature_extractor = ViTFeatureExtractor.from_pretrained(path1)
    model = ViTForImageClassification.from_pretrained(path2)
    classifier = VisionClassifierInference(feature_extractor=feature_extractor, model=model)

    label = classifier.predict(img_path=img_path)
    display_prediction(img_path, label)

def display_prediction(img_path, label):
    image = Image.open(img_path)
    image = image.resize((300, 300), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(image)

    label_image.config(image=photo)
    label_image.image = photo
    label_result.config(text="Predicted class: " + label)

# Paths
path1 = "./out/MYKVASIRV2MODEL/10_2024-03-22-00-08-07/feature_extractor/"
path2 = "./out/MYKVASIRV2MODEL/10_2024-03-22-00-08-07/model/"

# Tkinter setup
root = tk.Tk()
root.title("Image Classifier")

# Button to browse image
browse_button = tk.Button(root, text="Browse Image", command=browse_image)
browse_button.pack(pady=10)

# Label to display image
label_image = tk.Label(root)
label_image.pack()

# Label to display prediction
label_result = tk.Label(root, text="")
label_result.pack(pady=10)

root.mainloop()