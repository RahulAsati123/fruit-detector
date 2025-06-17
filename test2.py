# enhanced_fruit_detector.py
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

# Define the class BEFORE loading the pickle file
class QualcommAppleOrangeClassifier:
    def __init__(self, model_name="mobilenet_v2"):
        """Initialize with model"""
        self.base_model = models.mobilenet_v2(pretrained=True)
        self.modify_classifier()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.classes = ['apple', 'orange']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model.to(self.device)
    
    def modify_classifier(self):
        if hasattr(self.base_model, 'classifier'):
            in_features = self.base_model.classifier[-1].in_features
            self.base_model.classifier[-1] = nn.Linear(in_features, 2)
        elif hasattr(self.base_model, 'fc'):
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(in_features, 2)
    
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
        
        return {
            'predicted_class': self.classes[predicted_class],
            'confidence': confidence,
            'probabilities': {
                'apple': probabilities[0][0].item(),
                'orange': probabilities[0][1].item()
            }
        }

class BasketOrangeDetector:
    def __init__(self, model_path):
        self.device = torch.device('cpu')
        
        if model_path.endswith('.pkl'):
            with open(model_path, 'rb') as f:
                deployment_package = pickle.load(f)
                self.classifier = deployment_package['classifier']
                self.classes = deployment_package['classes']
        elif model_path.endswith('.pt'):
            checkpoint = torch.load(model_path, map_location='cpu')
            self.classes = checkpoint['classes']
            
            self.model = models.mobilenet_v2(pretrained=False)
            self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, 2)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict_single_crop(self, image_crop):
        if hasattr(self, 'classifier'):
            return self.classifier.predict_single_image(image_crop)
        else:
            input_tensor = self.transform(image_crop).unsqueeze(0)
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            return {
                'predicted_class': self.classes[predicted_class],
                'confidence': confidence,
                'probabilities': {
                    'apple': probabilities[0][0].item(),
                    'orange': probabilities[0][1].item()
                }
            }
    
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
    
    def detect_oranges_in_basket(self, image, confidence_threshold=0.6):
        crops, crop_info = self.create_overlapping_crops(image, grid_size=4, overlap=0.3)
        
        orange_detections = []
        all_predictions = []
        
        for i, (crop, info) in enumerate(zip(crops, crop_info)):
            try:
                prediction = self.predict_single_crop(crop)
                all_predictions.append(prediction)
                
                if (prediction['predicted_class'] == 'orange' and 
                    prediction['confidence'] > confidence_threshold):
                    
                    orange_detections.append({
                        'crop_id': i,
                        'position': info['position'],
                        'coordinates': info['coordinates'],
                        'center': info['center'],
                        'confidence': prediction['confidence'],
                        'orange_probability': prediction['probabilities']['orange']
                    })
            except Exception as e:
                st.warning(f"Error processing crop {i}: {e}")
        
        orange_confidences = [det['confidence'] for det in orange_detections]
        apple_confidences = [pred['probabilities']['apple'] for pred in all_predictions 
                           if pred['predicted_class'] == 'apple']
        
        total_crops = len(all_predictions)
        orange_crops = len(orange_detections)
        apple_crops = sum(1 for pred in all_predictions if pred['predicted_class'] == 'apple')
        
        has_orange = orange_crops > 0
        orange_percentage = (orange_crops / total_crops) * 100 if total_crops > 0 else 0
        
        return {
            'has_orange': has_orange,
            'orange_detections': orange_detections,
            'total_crops_analyzed': total_crops,
            'orange_crops_found': orange_crops,
            'apple_crops_found': apple_crops,
            'orange_percentage': orange_percentage,
            'average_orange_confidence': np.mean(orange_confidences) if orange_confidences else 0,
            'average_apple_confidence': np.mean(apple_confidences) if apple_confidences else 0,
            'all_predictions': all_predictions,
            'crop_info': crop_info
        }
    
    def visualize_detections(self, image, detection_results):
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)
        
        for detection in detection_results['orange_detections']:
            coords = detection['coordinates']
            confidence = detection['confidence']
            
            draw.rectangle(coords, outline='orange', width=3)
            
            label = f"Orange: {confidence:.2f}"
            try:
                font = ImageFont.load_default()
            except:
                font = None
            
            text_x = coords[0]
            text_y = coords[1] - 20 if coords[1] > 20 else coords[1]
            
            draw.text((text_x, text_y), label, fill='orange', font=font)
        
        return vis_image

# WebRTC configuration for camera
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

@st.cache_resource
def load_detector():
    try:
        model_path = "qualcomm_apple_orange_classifier.pkl"
        detector = BasketOrangeDetector(model_path)
        return detector
    except Exception as e:
        st.error(f"Error loading model: {e}")
        try:
            model_path = "qualcomm_apple_orange_classifier.pt"
            detector = BasketOrangeDetector(model_path)
            return detector
        except Exception as e2:
            st.error(f"Error loading .pt file: {e2}")
            return None

def main():
    st.set_page_config(
        page_title="Orange in Basket Detector",
        page_icon="🍊",
        layout="wide"
    )
    
    st.title("🍎🍊 Orange in Apple Basket Detector")
    st.write("**Enhanced Binary Classifier with Camera Support**")
    
    # Load detector
    detector = load_detector()
    if detector is None:
        st.error("Failed to load the model. Please check your model files.")
        st.stop()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Detection Settings")
    confidence_threshold = st.sidebar.slider(
        "Orange Confidence Threshold", 
        0.5, 0.95, 0.6, 0.05
    )
    
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
                
                if st.button("🔍 Analyze Basket", type="primary"):
                    with st.spinner("Analyzing fruit basket..."):
                        try:
                            results = detector.detect_oranges_in_basket(
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
            
            # Simple camera input
            camera_photo = st.camera_input("Take a photo of the fruit basket")
            
            if camera_photo is not None:
                image = Image.open(camera_photo).convert('RGB')
                st.image(image, caption="Captured Image", use_container_width=True)
                
                if st.button("🔍 Analyze Captured Image", type="primary"):
                    with st.spinner("Analyzing captured image..."):
                        try:
                            results = detector.detect_oranges_in_basket(
                                image, 
                                confidence_threshold=confidence_threshold
                            )
                            
                            st.session_state.results = results
                            st.session_state.original_image = image
                            st.session_state.analyzed = True
                            
                        except Exception as e:
                            st.error(f"Error during analysis: {e}")
        
        elif input_method == "Live Camera":
            st.header("📹 Live Camera Feed")
            st.info("Click on the camera feed to capture an image for analysis")
            
            # Initialize session state for captured frame
            if 'captured_frame' not in st.session_state:
                st.session_state.captured_frame = None
            
            # WebRTC camera stream
            class VideoProcessor:
                def __init__(self):
                    self.captured_frame = None
                
                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    
                    # Store the frame for analysis
                    st.session_state.latest_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Add text overlay
                    cv2.putText(img, "Click 'Capture Frame' to analyze", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            webrtc_ctx = webrtc_streamer(
                key="live-camera",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            
            # Capture button
            if st.button("📸 Capture Current Frame"):
                if 'latest_frame' in st.session_state:
                    st.session_state.captured_frame = st.session_state.latest_frame
                    st.success("Frame captured! Check the results panel.")
                else:
                    st.error("No frame available. Please ensure camera is active.")
            
            # Show captured frame
            if st.session_state.carried_frame is not None:
                st.image(st.session_state.captured_frame, caption="Captured Frame", use_container_width=True)
                
                if st.button("🔍 Analyze Captured Frame", type="primary"):
                    with st.spinner("Analyzing captured frame..."):
                        try:
                            image = Image.fromarray(st.session_state.captured_frame)
                            results = detector.detect_oranges_in_basket(
                                image, 
                                confidence_threshold=confidence_threshold
                            )
                            
                            st.session_state.results = results
                            st.session_state.original_image = image
                            st.session_state.analyzed = True
                            
                        except Exception as e:
                            st.error(f"Error during analysis: {e}")
    
    with col2:
        st.header("📊 Detection Results")
        
        if hasattr(st.session_state, 'analyzed') and st.session_state.analyzed:
            results = st.session_state.results
            original_image = st.session_state.original_image
            
            # Main result display
            if results['has_orange']:
                st.success("🍊 **ORANGES DETECTED IN BASKET!**")
                st.write(f"**Found oranges in {results['orange_crops_found']} out of {results['total_crops_analyzed']} analyzed regions**")
            else:
                st.info("🍎 **ONLY APPLES DETECTED**")
                st.write(f"**Analyzed {results['total_crops_analyzed']} regions, no oranges found**")
            
            # Statistics
            col_orange, col_apple, col_percent = st.columns(3)
            
            with col_orange:
                st.metric("🍊 Orange Regions", results['orange_crops_found'])
            
            with col_apple:
                st.metric("🍎 Apple Regions", results['apple_crops_found'])
            
            with col_percent:
                st.metric("🍊 Orange Coverage", f"{results['orange_percentage']:.1f}%")
            
            # Visualization
            if results['orange_detections']:
                st.write("### 🎯 Orange Detection Visualization")
                vis_image = detector.visualize_detections(original_image, results)
                st.image(vis_image, caption="Detected Orange Regions", use_container_width=True)
        
        else:
            st.info("Select an input method and analyze an image to see detection results")
    
    # Instructions
    with st.expander("ℹ️ How to Use Different Input Methods"):
        st.write("""
        ### 📤 Upload Image
        - Click 'Browse files' to upload an image from your device
        - Supports JPG, JPEG, PNG formats
        
        ### 📸 Camera Capture  
        - Click 'Take a photo' to capture using your device camera
        - Works on both desktop and mobile devices [[4]]
        
        ### 📹 Live Camera Feed
        - Real-time camera feed using WebRTC [[1]]
        - Click 'Capture Current Frame' to grab a frame for analysis
        - Requires camera permissions
        """)

if __name__ == "__main__":
    main()
