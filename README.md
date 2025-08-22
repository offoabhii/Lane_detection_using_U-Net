# 🚘 Lane Detection using U-Net

This project implements **road lane detection** using a U-Net deep learning architecture.  
It provides training code, prediction outputs, and an interactive **Gradio UI** for real-time testing on images, videos, and webcam feeds.  

---

## 📂 Project Structure
```
lane_detection_using_unet/
│── app.py                     # Gradio app for demo
│── final_project.ipynb         # Training & evaluation notebook
│── requirements.txt            # Dependencies
│── README.md                   # Project documentation
│── new_dataset/                # Training & prediction samples
│   ├── training/
│   ├── prediction/
│   └── prediction_refined/
│── old_dataset/                # Tu Simple lane prediction dataset
│   ├── TUSimple/
├         ├── train_set
├         ├── test_set
├         ├── test_label
│── .gitignore                  # Keeps repo clean
```

---

## 🔧 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/lane_detection_using_unet.git
cd lane_detection_using_unet
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Download pretrained weights
Download the trained U-Net weights from **GitHub Releases**:  
👉 [Download `unet_road_lane_best.h5`](https://github.com/your-username/lane_detection_using_unet/releases/latest)

Place the file in the **project root folder**.

---

## 🚀 Run the App
```bash
python app.py
```

This launches a Gradio web app with three modes:
- 🖼️ **Image Detection** → Upload an image
- 🎬 **Video Detection** → Upload a video
- 📹 **Live Webcam** → Real-time lane detection

---

## 🖥️ User Interface

Here’s the Gradio-based UI of the project:

![User Interface Screenshot](/ui.png)

*(Replace `assets/ui.png` with your actual screenshot path)*

---

## 🖼️ Prediction

Sample output of lane detection (green overlay = detected lanes):

![Prediction Example](/prediction_example.png)

---

## 📊 Results
- Model: **U-Net with skip connections**
- Input size: `128x256`
- Framework: **TensorFlow / Keras**
- Accuracy: Achieved stable lane segmentation on road datasets

---

## 🎥 Demo Video
👉 [Watch Demo Video](/lane_detection_presentation.mp4)

---

## 📌 Notes
- The model is trained on a custom lane dataset.
- Pretrained weights (`.h5`) and demo video (`.mp4`) are hosted under **Releases** to keep the repo lightweight.
- You can retrain the model using the `final_project.ipynb` notebook.

---

## 📜 License
This project is released under the **MIT License**.
