# enhanced_multi_fruit_detector.py
import streamlit as st
import torch
import torch.nn as nn
import pickle
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# Multi-Fruit Classifier for 10 different fruits
class MultiFruitClassifier:
    def __init__(self, model_name="mobilenet_v2", num_classes=10):
        """Initialize with model for multiple fruit classes"""
        self.base_model = models.mobilenet_v2(pretrained=True)
        self.num_classes = num_classes
        self.modify_classifier()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Extended fruit classes - 10 different fruits
        self.classes = [
            'apple', 'avocado', 'banana', 'cherry', 'kiwi', 
            'mango', 'orange', 'pineapple', 'strawberry', 'watermelon'
        ]
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model.to(self.device)
    
    def modify_classifier(self):
        if hasattr(self.base_model, 'classifier'):
            in_features = self.base_model.classifier[-1].in_features
            self.base_model.classifier[-1] = nn.Linear(in_features, self.num_classes)
        elif hasattr(self.base_model, 'fc'):
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(in_features, self.num_classes)
    
    def predict_single_image(self, image):
        self.base_model.eval()
        
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('RGB')
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.base_model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Create probability dictionary for all classes
        prob_dict = {}
        for i, class_name in enumerate(self.classes):
            prob_dict[class_name] = probabilities[0][i].item()
        
        return {
            'predicted_class': self.classes[predicted_class],
            'confidence': confidence,
            'probabilities': prob_dict
        }

class BasketMultiFruitDetector:
    def __init__(self, model_path=None, target_fruit='orange'):
        self.device = torch.device('cpu')
        self.target_fruit = target_fruit.lower()
        
        # Initialize with multi-fruit classifier
        if model_path and model_path.endswith('.pkl'):
            try:
                with open(model_path, 'rb') as f:
                    deployment_package = pickle.load(f)
                    self.classifier = deployment_package['classifier']
                    self.classes = deployment_package.get('classes', [
                        'apple', 'avocado', 'banana', 'cherry', 'kiwi', 
                        'mango', 'orange', 'pineapple', 'strawberry', 'watermelon'
                    ])
            except FileNotFoundError:
                st.warning(f"Model file {model_path} not found. Using default multi-fruit classifier.")
                self.classifier = MultiFruitClassifier(num_classes=10)
                self.classes = self.classifier.classes
        else:
            # Use default multi-fruit classifier
            self.classifier = MultiFruitClassifier(num_classes=10)
            self.classes = self.classifier.classes
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict_single_crop(self, image_crop):
        return self.classifier.predict_single_image(image_crop)
    
    def create_overlapping_crops(self, image, grid_size=4, overlap=0.3):
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        crops = []
        crop_info = []
        
        step_h = int(h * (1 - overlap) / (grid_size - 1))
        step_w = int(w * (1 - overlap) / (grid_size - 1))
        
        for i in range(grid_size):
            for j in range(grid_size):
                start_h = min(i * step_h, h - h//3)
                end_h = min(start_h + h//3, h)
                start_w = min(j * step_w, w - w//3)
                end_w = min(start_w + w//3, w)
                
                crop = img_array[start_h:end_h, start_w:end_w]
                crop_pil = Image.fromarray(crop)
                
                crops.append(crop_pil)
                crop_info.append({
                    'position': (i, j),
                    'coordinates': (start_w, start_h, end_w, end_h),
                    'center': ((start_w + end_w)//2, (start_h + end_h)//2)
                })
        
        return crops, crop_info
    
    def detect_fruits_in_basket(self, image, confidence_threshold=0.6):
        crops, crop_info = self.create_overlapping_crops(image, grid_size=4, overlap=0.3)
        
        target_detections = []
        all_predictions = []
        fruit_counts = {fruit: 0 for fruit in self.classes}
        fruit_confidences = {fruit: [] for fruit in self.classes}
        
        for i, (crop, info) in enumerate(zip(crops, crop_info)):
            try:
                prediction = self.predict_single_crop(crop)
                all_predictions.append(prediction)
                
                predicted_fruit = prediction['predicted_class']
                confidence = prediction['confidence']
                
                # Count all fruits detected with confidence above threshold
                if confidence > confidence_threshold:
                    fruit_counts[predicted_fruit] += 1
                    fruit_confidences[predicted_fruit].append(confidence)
                
                # Track target fruit detections
                if (predicted_fruit == self.target_fruit and 
                    confidence > confidence_threshold):
                    
                    target_detections.append({
                        'crop_id': i,
                        'position': info['position'],
                        'coordinates': info['coordinates'],
                        'center': info['center'],
                        'confidence': confidence,
                        'fruit_type': predicted_fruit,
                        'probability': prediction['probabilities'][predicted_fruit]
                    })
            except Exception as e:
                st.warning(f"Error processing crop {i}: {e}")
        
        total_crops = len(all_predictions)
        target_crops = len(target_detections)
        
        has_target_fruit = target_crops > 0
        target_percentage = (target_crops / total_crops) * 100 if total_crops > 0 else 0
        
        # Calculate average confidences for each fruit type
        avg_confidences = {}
        for fruit in self.classes:
            if fruit_confidences[fruit]:
                avg_confidences[fruit] = np.mean(fruit_confidences[fruit])
            else:
                avg_confidences[fruit] = 0.0
        
        return {
            'has_target_fruit': has_target_fruit,
            'target_fruit': self.target_fruit,
            'target_detections': target_detections,
            'total_crops_analyzed': total_crops,
            'target_crops_found': target_crops,
            'target_percentage': target_percentage,
            'fruit_counts': fruit_counts,
            'average_confidences': avg_confidences,
            'all_predictions': all_predictions,
            'crop_info': crop_info
        }
    
    def visualize_detections(self, image, detection_results):
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)
        
        # Color mapping for different fruits
        fruit_colors = {
            'apple': 'red', 'avocado': 'green', 'banana': 'yellow',
            'cherry': 'darkred', 'kiwi': 'brown', 'mango': 'orange',
            'orange': 'orange', 'pineapple': 'gold', 'strawberry': 'pink',
            'watermelon': 'lightgreen'
        }
        
        # Draw all detected fruits, not just target fruit
        all_detections = []
        
        # Get all high-confidence detections
        for i, (crop_info, prediction) in enumerate(zip(detection_results['crop_info'], detection_results['all_predictions'])):
            if prediction['confidence'] > 0.6:  # Use same threshold
                all_detections.append({
                    'coordinates': crop_info['coordinates'],
                    'confidence': prediction['confidence'],
                    'fruit_type': prediction['predicted_class']
                })
        
        for detection in all_detections:
            coords = detection['coordinates']
            confidence = detection['confidence']
            fruit_type = detection['fruit_type']
            
            color = fruit_colors.get(fruit_type, 'blue')
            
            # Draw rectangle
            draw.rectangle(coords, outline=color, width=3)
            
            # Add label
            label = f"{fruit_type.title()}: {confidence:.2f}"
            try:
                font = ImageFont.load_default()
            except:
                font = None
            
            text_x = coords[0]
            text_y = coords[1] - 20 if coords[1] > 20 else coords[1]
            
            draw.text((text_x, text_y), label, fill=color, font=font)
        
        return vis_image

# Enhanced WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {
                "urls": ["turn:openrelay.metered.ca:80"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
        ]
    }
)

@st.cache_resource
def load_detector(target_fruit):
    try:
        detector = BasketMultiFruitDetector(target_fruit=target_fruit)
        return detector
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    st.set_page_config(
        page_title="Multi-Fruit Detector",
        page_icon="🍎",
        layout="wide"
    )
    
    st.title("🍎🍊🥭 Multi-Fruit Basket Detector")
    st.write("**Advanced Classifier Supporting 10 Different Fruits with Fixed Camera**")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Detection Settings")
    
    # Fruit selection
    available_fruits = [
        'apple', 'avocado', 'banana', 'cherry', 'kiwi', 
        'mango', 'orange', 'pineapple', 'strawberry', 'watermelon'
    ]
    
    target_fruit = st.sidebar.selectbox(
        "Select target fruit to detect:",
        available_fruits,
        index=6  # Default to orange
    )
    
    confidence_threshold = st.sidebar.slider(
        f"{target_fruit.title()} Confidence Threshold", 
        0.5, 0.95, 0.6, 0.05
    )
    
    # Load detector with selected fruit
    detector = load_detector(target_fruit)
    if detector is None:
        st.error("Failed to load the model.")
        st.stop()
    
    # Input method selection
    st.sidebar.header("📷 Input Method")
    input_method = st.sidebar.selectbox(
        "Choose input method:",
        ["Upload Image", "Camera Capture", "Live Camera"]
    )
    
    # Main layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if input_method == "Upload Image":
            st.header("📤 Upload Fruit Basket Image")
            
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=["jpg", "jpeg", "png"]
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                if st.button(f"🔍 Detect {target_fruit.title()}", type="primary"):
                    with st.spinner(f"Analyzing for {target_fruit}..."):
                        try:
                            results = detector.detect_fruits_in_basket(
                                image, 
                                confidence_threshold=confidence_threshold
                            )
                            
                            st.session_state.results = results
                            st.session_state.original_image = image
                            st.session_state.analyzed = True
                            
                        except Exception as e:
                            st.error(f"Error during analysis: {e}")
        
        elif input_method == "Camera Capture":
            st.header("📸 Camera Capture")
            
            try:
                camera_photo = st.camera_input("Take a photo of the fruit basket")
                
                if camera_photo is not None:
                    image = Image.open(camera_photo).convert('RGB')
                    st.image(image, caption="Captured Image", use_container_width=True)
                    
                    if st.button(f"🔍 Detect {target_fruit.title()}", type="primary"):
                        with st.spinner(f"Analyzing for {target_fruit}..."):
                            try:
                                results = detector.detect_fruits_in_basket(
                                    image, 
                                    confidence_threshold=confidence_threshold
                                )
                                
                                st.session_state.results = results
                                st.session_state.original_image = image
                                st.session_state.analyzed = True
                                
                            except Exception as e:
                                st.error(f"Error during analysis: {e}")
            except Exception as e:
                st.error(f"Camera access error: {e}")
                st.info("Please ensure your browser has camera permissions enabled.")
        
        elif input_method == "Live Camera":
            st.header("📹 Live Camera Feed")
            st.info("Real-time camera feed - Click 'Capture Frame' to analyze")
            
            # Initialize session state
            if 'captured_frame' not in st.session_state:
                st.session_state.captured_frame = None
            if 'latest_frame' not in st.session_state:
                st.session_state.latest_frame = None
            
            # Fixed WebRTC implementation
            class VideoProcessor:
                def __init__(self):
                    self.frame_count = 0
                
                def recv(self, frame):
                    self.frame_count += 1
                    img = frame.to_ndarray(format="bgr24")
                    
                    # Store the latest frame
                    if self.frame_count % 10 == 0:
                        st.session_state.latest_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Add overlay text
                    cv2.putText(img, "Live Multi-Fruit Detection Feed", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(img, f"Target: {target_fruit.title()}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            try:
                webrtc_ctx = webrtc_streamer(
                    key="multi-fruit-camera",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=RTC_CONFIGURATION,
                    video_processor_factory=VideoProcessor,
                    media_stream_constraints={
                        "video": {
                            "width": {"min": 640, "ideal": 1280, "max": 1920},
                            "height": {"min": 480, "ideal": 720, "max": 1080},
                        },
                        "audio": False
                    },
                    async_processing=True,
                )
                
                # Capture controls
                col_capture, col_status = st.columns([1, 2])
                
                with col_capture:
                    capture_clicked = st.button("📸 Capture Current Frame", type="primary")
                
                with col_status:
                    if webrtc_ctx.state.playing:
                        st.success("🟢 Camera Active")
                    else:
                        st.warning("🔴 Camera Not Active")
                
                # Handle frame capture - FIXED the typo here
                if capture_clicked:
                    if st.session_state.latest_frame is not None:
                        st.session_state.captured_frame = st.session_state.latest_frame.copy()
                        st.success("✅ Frame captured successfully!")
                    else:
                        st.error("❌ No frame available. Please ensure camera is active.")
                
                # Show captured frame and analysis
                if st.session_state.captured_frame is not None:
                    st.write("### 📸 Captured Frame")
                    st.image(st.session_state.captured_frame, caption="Captured Frame", use_container_width=True)
                    
                    if st.button(f"🔍 Analyze for {target_fruit.title()}", type="primary"):
                        with st.spinner(f"Analyzing for {target_fruit}..."):
                            try:
                                image = Image.fromarray(st.session_state.captured_frame)
                                results = detector.detect_fruits_in_basket(
                                    image, 
                                    confidence_threshold=confidence_threshold
                                )
                                
                                st.session_state.results = results
                                st.session_state.original_image = image
                                st.session_state.analyzed = True
                                
                            except Exception as e:
                                st.error(f"Error during analysis: {e}")
                
            except Exception as e:
                st.error(f"WebRTC Error: {e}")
                st.info("Live camera may not work in all environments. Try 'Camera Capture' instead.")
    
    with col2:
        st.header("📊 Multi-Fruit Detection Results")
        
        if hasattr(st.session_state, 'analyzed') and st.session_state.analyzed:
            results = st.session_state.results
            original_image = st.session_state.original_image
            
            # Main result display
            if results['has_target_fruit']:
                st.success(f"🎯 **{results['target_fruit'].upper()} DETECTED IN BASKET!**")
                st.write(f"**Found {results['target_fruit']} in {results['target_crops_found']} out of {results['total_crops_analyzed']} analyzed regions**")
            else:
                st.info(f"❌ **NO {results['target_fruit'].upper()} DETECTED**")
                st.write(f"**Analyzed {results['total_crops_analyzed']} regions, no {results['target_fruit']} found**")
            
            # All fruits distribution
            st.write("### 🍎 All Fruits Distribution in Basket")
            fruit_counts = results['fruit_counts']
            
            # Display fruit counts in a grid
            cols = st.columns(5)
            col_idx = 0
            for fruit, count in fruit_counts.items():
                if count > 0:
                    with cols[col_idx % 5]:
                        emoji_map = {
                            'apple': '🍎', 'avocado': '🥑', 'banana': '🍌',
                            'cherry': '🍒', 'kiwi': '🥝', 'mango': '🥭',
                            'orange': '🍊', 'pineapple': '🍍', 'strawberry': '🍓',
                            'watermelon': '🍉'
                        }
                        emoji = emoji_map.get(fruit, '🍇')
                        st.metric(f"{emoji} {fruit.title()}", count)
                        col_idx += 1
            
            # Target fruit statistics
            col_target, col_total, col_percent = st.columns(3)
            
            with col_target:
                st.metric(f"🎯 {results['target_fruit'].title()} Regions", results['target_crops_found'])
            
            with col_total:
                st.metric("📊 Total Regions", results['total_crops_analyzed'])
            
            with col_percent:
                st.metric(f"📈 {results['target_fruit'].title()} Coverage", f"{results['target_percentage']:.1f}%")
            
            # Confidence scores for detected fruits
            st.write("### 📈 Average Confidence Scores")
            avg_confidences = results['average_confidences']
            detected_fruits = {k: v for k, v in avg_confidences.items() if v > 0}
            
            if detected_fruits:
                conf_cols = st.columns(min(len(detected_fruits), 4))
                for i, (fruit, confidence) in enumerate(detected_fruits.items()):
                    with conf_cols[i % 4]:
                        st.metric(f"{fruit.title()} Confidence", f"{confidence:.2f}")
            
            # Visualization with all fruits
            st.write("### 🎯 Multi-Fruit Detection Visualization")
            vis_image = detector.visualize_detections(original_image, results)
            st.image(vis_image, caption="All Detected Fruits", use_container_width=True)
        
        else:
            st.info("Select a target fruit and analyze an image to see detection results")
    
    # Enhanced instructions
    with st.expander("ℹ️ Multi-Fruit Detector Guide"):
        st.write(f"""
        ### 🎯 Supported Fruits
        This detector can identify **10 different fruits**:
        🍎 Apple, 🥑 Avocado, 🍌 Banana, 🍒 Cherry, 🥝 Kiwi, 
        🥭 Mango, 🍊 Orange, 🍍 Pineapple, 🍓 Strawberry, 🍉 Watermelon
        
        ### 🔧 How It Works
        - **Grid Analysis**: Divides images into overlapping regions for thorough detection
        - **Multi-Class Recognition**: Identifies all fruits simultaneously 
        - **Target Focus**: Highlights your selected target fruit while showing all detected fruits
        - **Confidence Scoring**: Shows detection confidence for each fruit type
        
        ### 📤 Input Methods
        - **Upload Image**: Most reliable, supports JPG/PNG formats
        - **Camera Capture**: Direct photo capture, works on mobile and desktop
        - **Live Camera**: Real-time feed with frame capture (may need permissions)
        
        ### 🎨 Visualization
        - Different colored bounding boxes for each fruit type
        - Confidence scores displayed on each detection
        - Complete fruit distribution statistics
        """)

if __name__ == "__main__":
    main()
