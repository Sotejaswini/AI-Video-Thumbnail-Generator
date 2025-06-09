# AI Video Thumbnail Generator

## Overview

The **AI Video Thumbnail Generator** is an AI-powered tool that processes videos to generate visually appealing thumbnails. It uses computer vision techniques such as motion detection, object and face detection (via YOLOv8), auto-cropping, and text overlays to create thumbnails in multiple aspect ratios. The project also includes scripts to extract original frames and generate side-by-side comparisons for analysis.

### Video Explanation
Watch a detailed explanation of the approach, implementation, and results:

[🔗 Watch the Demo Video](https://drive.google.com/file/d/19vVmQrljInf9PT6pCNu1iATCcEmjod6h/view?usp=sharing)

This project demonstrates skills in Python, OpenCV, PIL, YOLO, and debugging in a WSL environment.

### Key Highlights:
- Processes **5 videos**
- Generates **15 thumbnails** (3 aspect ratios per video)
- Extracts **5 original frames**
- Creates **5 comparison images**

---

## Features

### Thumbnail Generation
- Key frame extraction using motion detection
- Face and object detection with YOLOv8
- Auto-cropping to focus on detected entities
- Text overlays with customizable messages
- Thumbnails in **16:9**, **4:3**, and **1:1** aspect ratios

###  Original Frame Extraction
- Extracts the **middle frame** of each video as a representative original

### Comparison Generation
- Creates side-by-side comparisons of the original frame and the 16:9 thumbnail

---
## Project Structure
```
├── videos/ # Input videos (e.g., video1.mp4)
├── thumbnails/ # Generated thumbnails
├── original_frames/ # Extracted original frames
├── comparisons/ # Side-by-side comparison images
├── thumbnail_generator.py # Main script for thumbnail generation
├── extract_frames.py # Extracts original frames
├── create_comparisons.py # Creates comparison images
├── ariblk.ttf # Font file for text overlay
├── yolov8n.pt # YOLOv8 model weights
├── requirements.txt # Dependencies
└── README.md # Project documentation
```
---

## Prerequisites

### System Requirements
- Python 3.8 or higher
- FFmpeg (for OpenCV-compatible video handling)

### Environment
Developed in **WSL (Windows Subsystem for Linux)**. A Linux environment is recommended for optimal compatibility.

---
## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/[your-username]/[your-repo-name].git
cd [your-repo-name]
```
2. Set Up Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```
4. Install Python Dependencies
```bash
pip install -r requirements.txt
```
5. Verify YOLO Model
Ensure yolov8n.pt is present in the project directory. This is the YOLOv8 weights file for detection.

Usage
1. Generate Thumbnails
Single Video:

```bash
python3 thumbnail_generator.py --video videos/video1_converted.mp4 --text "Watch Now!"
```
Batch Mode (All Videos):
```bash
python3 thumbnail_generator.py --video videos --text "Click to Watch!" --batch
```
2. Extract Original Frames
```bash
python3 extract_frames.py
```
3. Create Comparison Images
```bash
python3 create_comparisons.py
```
Example Output
```
Input Video: videos/video1_converted.mp4

Generated Thumbnails:
thumbnails/video1_16x9_20250608_211530.jpg
thumbnails/video1_4x3_20250608_211530.jpg
thumbnails/video1_1x1_20250608_211530.jpg

Original Frame:
original_frames/video1.mp4_original.jpg

Comparison Image:
comparisons/video1_comparison.jpg
```
Challenges & Solutions
```
WSL Video Access Issue
Problem: OpenCV failed to read certain videos
Solution: Used FFmpeg to re-encode videos for compatibility
```
Deprecated PIL Method
```
Problem: textsize() deprecated in newer PIL versions
Solution: Replaced with textbbox() for text dimension calculation
```
Filename Mismatch
```
Problem: Inconsistent naming between thumbnails and original frames
Solution: Standardized naming conventions in thumbnail_generator.py and fixed mapping logic in create_comparisons.py
```
Dependencies
```
See requirements.txt for the full list. Key libraries include:
opencv-python: Video and frame processing
Pillow: Image manipulation and text overlays
ultralytics: YOLOv8 for object detection
numpy: Array operations
```

