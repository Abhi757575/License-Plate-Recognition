# 🚗 License Plate Recognition (LPR) System

A proof-of-concept **License Plate Recognition (LPR)** pipeline that combines **YOLOv8 vehicle detection**, **SORT multi-object tracking**, and **EasyOCR-based text recognition** to detect, track, and read license plates from video streams.

---

## 📌 Overview

This project processes video frames to:

1. Detect vehicles (cars, bikes, buses, trucks)
2. Localize license plates within those vehicles
3. Track vehicles across frames
4. Extract and normalize license plate text using OCR

---

## 🚀 Key Features

- 🔍 **Vehicle Detection**  
  Uses **YOLOv8 (`yolov8n.pt`)** trained on COCO dataset to detect:
  - Cars
  - Motorcycles
  - Buses
  - Trucks

- 🔲 **License Plate Detection**  
  Custom-trained model (`models/license_plate_detector.pt`) to detect plate regions inside vehicles.

- 🔄 **Object Tracking**  
  Implements **SORT (Simple Online Realtime Tracking)** to assign consistent IDs (`car_id`) across frames.

- 🔤 **OCR + Text Normalization**  
  Uses **EasyOCR** to read plate text and applies heuristic corrections:
  - `0 ↔ O`
  - `1 ↔ I`
  - `5 ↔ S`
  - etc.

---

## 🏗️ Project Structure

License-Plate-Recognition/
│── models/ # Trained models (not included in repo)
│── videos/ # Input test videos (ignored in git)
│── output/ # Output annotated videos
│── src/ # Core source code
│── notebooks/ # Colab / experimentation notebooks
│── requirements.txt
│── README.md
│── .gitignore





---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/Abhi757575/License-Plate-Recognition.git
cd License-Plate-Recognition


2. Install dependencies
pip install -r requirements.txt
3. Install additional tools
pip install ultralytics easyocr filterpy
▶️ Usage
Run the pipeline
python src/main.py
Input
Video file (.mp4)
Frame stream
Output
Annotated video with:
Bounding boxes (vehicles + plates)
Tracking IDs
Extracted license plate text
🧠 Pipeline Workflow
Video → Frame Extraction
      → YOLOv8 Vehicle Detection
      → License Plate Detection
      → SORT Tracking
      → Plate Cropping
      → OCR (EasyOCR)
      → Text Normalization
      → Output Rendering
⚠️ Notes
Large files like:
.mp4 videos
.pt models
are not included in this repo due to GitHub size limits.

👉 Add your own:

yolov8n.pt
license_plate_detector.pt
Input videos
📊 Future Improvements
✅ Improve OCR accuracy with custom training
✅ Add real-time webcam support
✅ Deploy as web app (Flask / FastAPI)
✅ Use DeepSORT for better tracking
✅ Integrate database for plate logging
🤝 Contributing

Contributions are welcome! Feel free to:

Open issues
Submit pull requests
Suggest improvements
📜 License

This project is for educational and research purposes.

🙌 Acknowledgements
YOLOv8 by Ultralytics
SORT Tracking Algorithm
EasyOCR
