# 🚗 Car Damage Detection System using YOLOv8

This project is a **Streamlit web application** powered by a **YOLOv8 object detection model** trained to detect damaged car parts from uploaded images.

It can identify and localize different types of damages like dents and scratches on specific parts of a vehicle such as the front bumper, hood, door, and more.

---

## 🎯 Features

- 🧠 Trained YOLOv8s model with 5 custom classes
- 🌐 Interactive Streamlit web interface
- 📦 Upload and detect car part damage in real-time
- 📊 Detection summary with class names and confidence scores

---

## 🧠 Model Training

This model was trained on a labeled dataset using the following command:

```bash
yolo task=detect mode=train model=yolov8s.pt data="C:/Users/toffe/OneDrive/Documents/KUSH_DINE/New folder/CODE/image_dataset/car_damage.yaml" epochs=50 imgsz=640
```

### 🏷️ Classes:

```yaml
names:
  0: headlamp
  1: front_bumper
  2: hood
  3: door
  4: rear_bumper
```

The final model checkpoint was saved to:
```
runs/detect/train4/weights/best.pt
```

---

## 📂 Dataset

- **Name:** Coco Car Damage Detection Dataset  
- **Source:** [Kaggle - lplenka/coco-car-damage-detection-dataset](https://www.kaggle.com/datasets/lplenka/coco-car-damage-detection-dataset)
- **Format:** COCO JSON annotations
- **Usage:** Converted to YOLO format using `coco2yolo.py`

---

## 🖥️ Live App Demo

### 🔎 Input Image
![Input](assets/input.jpg)

### ✅ Output with Detection
![Output](assets/output.jpg)

---

## 🗂️ Project Structure

```
CODE/
├── app.py                   # Streamlit app for detection
├── coco2yolo.py             # Script to convert COCO annotations to YOLO format
├── yolov8s.pt               # Pretrained YOLO model (before fine-tuning)
├── image_dataset/           # Dataset and config
│   ├── images/train/
│   ├── images/val/
│   ├── labels/train/
│   ├── labels/val/
│   └── car_damage.yaml
├── runs/detect/train4/      # YOLO training output
│   └── weights/best.pt      # Trained model weights
├── assets/                  # Screenshots and visual outputs
│   ├── input.jpg
│   └── output.jpg
```

---

## ▶️ How to Run the Project

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

or manually:

```bash
pip install streamlit ultralytics pillow
```

### 2. Run Streamlit App

```bash
streamlit run app.py
```

Then open the local server link shown in the terminal (usually `http://localhost:8501`).

---

## 🧪 Sample Usage

- Drag and drop a `.jpg` or `.png` car image.
- The model returns a prediction image with bounding boxes and confidence scores.
- Damaged parts like `front_bumper` or `hood` are detected automatically.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Priya Barman**  
_M.Tech AI_

---

## 🌐 Related Tools

- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [Streamlit Docs](https://docs.streamlit.io/)
