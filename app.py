import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import os
import tempfile

# Import custom modules
from ultralytics import YOLO
from Text_recognition.CRNN import CRNN
from rcnn_preprocessing import main as get_preprocessing_data, decode

# Configure page
st.set_page_config(
    page_title="OCR Text Recognition System",
    layout="wide"
)

def load_models():
    """Load both YOLO and CRNN models"""
    try:
        # Load YOLO model
        yolo_model_path = "runs/detect/train/weights/best.pt"
        if not os.path.exists(yolo_model_path):
            st.error(f"YOLO model not found at {yolo_model_path}")
            return None, None, None, None
        
        yolo_model = YOLO(yolo_model_path)
        
        # Load CRNN model
        crnn_model_path = "crnn/best_model.pt"
        if not os.path.exists(crnn_model_path):
            st.error(f"CRNN model not found at {crnn_model_path}")
            return None, None, None, None
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(crnn_model_path, map_location=torch.device(device))
        
        crnn_model = CRNN(
            vocab_size=checkpoint['vocab_size'],
            hidden_size=checkpoint['hidden_size'],
            n_layers=checkpoint['n_layers'],
            dropout=checkpoint['dropout_prob']
        ).to(device)
        crnn_model.load_state_dict(checkpoint['model_state_dict'])
        crnn_model.eval()
        
        # Load preprocessing data
        preprocessing_data = get_preprocessing_data()
        data_transforms = preprocessing_data['data_transforms']['val']
        idx_to_char = preprocessing_data['idx_to_char']
        
        return yolo_model, crnn_model, data_transforms, idx_to_char
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

def process_image(image, yolo_model, crnn_model, data_transforms, idx_to_char, confidence_threshold=0.5):
    """Process image with both YOLO and CRNN models"""
    
    # Convert PIL to cv2 format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # YOLO detection
    yolo_results = yolo_model(image_cv)
    
    detected_texts = []
    annotated_image = image_cv.copy()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for result in yolo_results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        
        for i, (box, score) in enumerate(zip(boxes, scores)):
            if score > confidence_threshold:
                x1, y1, x2, y2 = map(int, box)
                
                # Crop the detected region
                cropped_region = image_cv[y1:y2, x1:x2]
                
                if cropped_region.size > 0:
                    # Convert to PIL Image and apply transforms
                    pil_image = Image.fromarray(cv2.cvtColor(cropped_region, cv2.COLOR_BGR2RGB))
                    transformed_image = data_transforms(pil_image).unsqueeze(0).to(device)
                    
                    # Predict text using CRNN
                    with torch.no_grad():
                        output = crnn_model(transformed_image)
                        output = output.permute(1, 0, 2)  # Convert back from CTC format
                        
                        # Get predicted sequence (greedy decoding)
                        predicted_sequence = torch.argmax(output, dim=2)
                        decoded_text = decode(predicted_sequence, idx_to_char)[0]
                    
                    # Store result
                    detected_texts.append({
                        'text': decoded_text,
                        'confidence': score,
                        'box': (x1, y1, x2, y2)
                    })
                    
                    # Draw bounding box and text on image
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{decoded_text} ({score:.2f})"
                    cv2.putText(annotated_image, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Convert back to RGB for display
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    return detected_texts, annotated_image_rgb

def main():
    st.title("OCR Text Recognition System")
    st.write("Upload an image to detect and recognize text using YOLOv11 + CRNN")
    
    # Load models
    with st.spinner("Loading models..."):
        yolo_model, crnn_model, data_transforms, idx_to_char = load_models()
    
    if not all([yolo_model, crnn_model, data_transforms, idx_to_char]):
        st.error("Failed to load models. Please check if model files exist.")
        st.info("Required files: `runs/detect/train/weights/best.pt` and `crnn/best_model.pt`")
        return
    
    st.success("Models loaded successfully!")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff']
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        # Tự động nhận dạng ngay sau khi tải ảnh
        default_threshold = 0.5
        with st.spinner("Đang nhận dạng văn bản..."):
            detected_texts, annotated_image = process_image(
                image,
                yolo_model,
                crnn_model,
                data_transforms,
                idx_to_char,
                confidence_threshold=default_threshold
            )
        
        # Hiển thị ảnh gốc và kết quả
        st.subheader("Ảnh gốc")
        st.image(image, caption="Ảnh đã tải lên", use_column_width=True)
        
        st.subheader("Kết quả nhận dạng")
        st.image(annotated_image, caption="Vùng văn bản được phát hiện", use_column_width=True)
        
        # Hiển thị văn bản đã nhận dạng
        st.subheader("Văn bản")
        if detected_texts:
            for i, result in enumerate(detected_texts):
                with st.expander(f"Vùng {i+1}: '{result['text']}'"):
                    st.write(f"**Text:** {result['text']}")
                    st.write(f"**Confidence:** {result['confidence']:.3f}")
                    st.write(f"**Bounding Box:** {result['box']}")
            
            # Tổng hợp
            st.subheader("Tổng hợp")
            all_text = " ".join([result['text'] for result in detected_texts if result['text'].strip()])
            st.write(f"**Số vùng phát hiện:** {len(detected_texts)}")
            st.write(f"**Chuỗi văn bản ghép:** {all_text}")
        else:
            st.warning("Không phát hiện văn bản.")

if __name__ == "__main__":
    main() 