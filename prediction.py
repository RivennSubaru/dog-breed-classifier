import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np

img_size = (256, 256)
classes = ['doberman', 'chihuahua', 'chien_nu_du_perou']
model_path = "model_chien.h5"
image_path = "data/test/chien_nu_du_perou/9XYMXQRE72QG.jpg"

model = tf.keras.models.load_model(model_path)

def predict_and_show(image_path):
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_name = classes[class_index]
    confidence = prediction[0][class_index]

    plt.imshow(img)
    plt.title(f"Prédiction : {class_name} ({confidence*100:.2f}%)", fontsize=14)
    plt.axis('off')
    plt.show()

predict_and_show(image_path)
