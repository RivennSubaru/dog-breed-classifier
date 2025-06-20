# 🐶 Dog Classifier (Doberman, Chihuahua, Peruvian Hairless)

### *Created by ZAFIMENA Richard Steven | 140I23*

This project is a deep learning model built with TensorFlow to classify dog breeds from images. The model can identify:

* Doberman
* Chihuahua
* Peruvian Hairless Dog

## 📁 Project Structure

```
Dog Classifier/
├── data/
│   └── train/
│   └── test/
├── model_chien.h5
├── main.py
├── prediction.py
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 How to Use

### 1. Train the Model

Place your training images in the following structure:

```
data/train/
├── doberman/
├── chihuahua/
└── chien_nu_du_perru/
```

Then run the training script:

```bash
python main.py
```

The trained model will be saved to `model_chien.h5`.

---

### 2. Predict an Image

Put a test image anywhere and update the path in `prediction.py` (variable `image_path`). Then run:

```bash
python prediction.py
```

A window will open showing the image and the predicted breed on top.

---

## ✅ Requirements

* Python 3.8+
* TensorFlow
* matplotlib
* numpy
* Pillow

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

##
