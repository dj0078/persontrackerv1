# persontrackerv1

!!! python 3.10 !!!

**personextractorv1** is a real-time computer vision tool that detects and tracks people in video using YOLOv8 and DeepSORT. It annotates each detected person with a unique ID while maintaining clean and minimal visual output, suitable for further analytics or video processing pipelines.

---

## 🔧 Features

- ✅ Person detection using **YOLOv8**
- 🔁 Person tracking using **DeepSORT**
- 🟩 Minimal green bounding boxes (no clutter)
- 🆔 Track IDs displayed for each person
- 🧼 Clean frame handling (no annotation overlap)
- 🎯 Accurate bounding box refinement with IoU matching
- 📦 Ready for integration with logging/export/face blurring systems

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/personextractorv1.git
cd personextractorv1

2. Install Dependencies
pip install -r requirements.txt
(manual) pip install ultralytics deep_sort_realtime opencv-python

3. Usage 
1) Add video
2)run Script py -3.10 main.py

4. Project Structure
personextractorv1/
├── data/
│   └── SampleVideo1.mp4          # Your input video
├── main.py                       # Main tracking script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── .gitignore                    # Git ignored files

Created by JEONG DONGWOON 2025