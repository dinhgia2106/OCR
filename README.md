# OCR system

## Datasets

Install from [here](https://drive.google.com/file/d/1kUy2tuH-kKBlFCNA0a9sqD2TG4uyvBnV/view)

## Pipeline

Image -> text detection ('YOLO') -> text recognition ('CRNN') -> text

## Preprocessing

Run `xml_to_yolo.py` to convert data to YOLO format

## Train YOLOv11

Run `python -m Text_detection.YOLOv11`.
