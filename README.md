# OCR System Overview

## Datasets

**Download Dataset**
Access and download the dataset from this [Google Drive link](https://drive.google.com/file/d/1kUy2tuH-kKBlFCNA0a9sqD2TG4uyvBnV/view).

---

## Processing Pipeline

```plaintext
Input Image → Text Detection (YOLOv11) → Text Recognition (CRNN) → Final Text Output
```

---

## Phase 1: Text Detection

### Preprocessing

Convert XML annotations to YOLO format:

```bash
python xml_to_yolo.py
```

### Training YOLOv11

Start training the YOLOv11 detection model:

```bash
python -m Text_detection.YOLOv11
```

---

## Phase 2: Text Recognition

### Preprocessing

Prepare image–text pairs for training:

```bash
python prepare_crnn_data.py
```

---

_Note:_ This script processes the dataset into a format compatible with CRNN-based text recognition (image + corresponding label only).

### Training CRNN

Start training the CRNN model:

```bash
python -m Text_recognition.CRNN
```
