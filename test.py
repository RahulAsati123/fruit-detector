# enhanced_fruit_detector_modern_ui.py
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
import time

# Custom CSS for modern styling
def load_custom_css():
    st.markdown("""
    <style>
    /* Main app styling */
    .main {
        padding-top: 2rem;
    }
    
    /* Custom header styling */
    .custom-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .custom-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .custom-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Card styling */
    .custom-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid #e1e5e9;
        margin-bottom: 2rem;
        transition: transform 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .metric-card p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Fruit emoji styling */
    .fruit-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .fruit-item {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .fruit-item:hover {
        border-color: #667eea;
        transform: scale(1.05);
    }
    
    .fruit-emoji {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    /* Status indicators */
    .status-success {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        font-weight: 600;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Progress bar styling */
    .stProgress .st-bo {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* File uploader styling */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9ff;
    }
    
    /* Camera styling */
    .camera-container {
        border: 3px solid #667eea;
        border-radius: 15px;
        padding: 1rem;
        background: #f8f9ff;
    }
    </style>
    """, unsafe_allow_html=True)

# Multi-Fruit Classifier (same as before)
class MultiFruitClassifier:
    def __init__(self, model_name="mobilenet_v2", num_classes=10):
        self.base_model = models.mobilenet_v2(pretrained=True)
        self.num_classes = num_classes
        self.modify_classifier()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
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
        
        prob_dict = {}
        for i, class_name in enumerate(self.classes):
            prob_dict[class_name] = probabilities[0][i].item()
        
        return {
            'predicted_class': self.classes[predicted_class],
            'confidence': confidence,
            'probabilities': prob_dict
        }

# Detector class (simplified for brevity - use your existing logic)
class BasketMultiFruitDetector:
    def __init__(self, target_fruit='orange'):
        self.target_fruit = target_fruit.lower()
        self.classifier = MultiFruitClassifier(num_classes=10)
        self.classes = self.classifier.classes
    
    def detect_fruits_in_basket(self, image, confidence_threshold=0.6):
        # Simplified detection for demo - replace with your actual logic
        prediction = self.classifier.predict_single_image(image)
        
        # Mock results for demonstration
        fruit_counts = {fruit: 0 for fruit in self.classes}
        if prediction['confidence'] > confidence_threshold:
            fruit_counts[prediction['predicted_class']] = 1
        
        has_target = prediction['predicted_class'] == self.target_fruit and prediction['confidence'] > confidence_threshold
        
        return {
            'has_target_fruit': has_target,
            'target_fruit': self.target_fruit,
            'fruit_counts': fruit_counts,
            'confidence': prediction['confidence'],
            'predicted_fruit': prediction['predicted_class']
        }

@st.cache_resource
def load_detector(target_fruit):
    return BasketMultiFruitDetector(target_fruit=target_fruit)

def create_modern_header():
    st.markdown("""
    <div class="custom-header">
        <h1>🍎🍊🥭 AI Fruit Detector</h1>
        <p>Advanced Multi-Fruit Recognition System with Modern UI</p>
    </div>
    """, unsafe_allow_html=True)

def create_fruit_selector():
    st.markdown("### 🎯 Select Target Fruit")
    
    fruit_options = {
        '🍎 Apple': 'apple',
        '🥑 Avocado': 'avocado', 
        '🍌 Banana': 'banana',
        '🍒 Cherry': 'cherry',
        '🥝 Kiwi': 'kiwi',
        '🥭 Mango': 'mango',
        '🍊 Orange': 'orange',
        '🍍 Pineapple': 'pineapple',
        '🍓 Strawberry': 'strawberry',
        '🍉 Watermelon': 'watermelon'
    }
    
    selected = st.selectbox(
        "Choose the fruit you want to detect:",
        list(fruit_options.keys()),
        index=6  # Default to orange
    )
    
    return fruit_options[selected]

def create_confidence_slider(target_fruit):
    st.markdown("### ⚙️ Detection Settings")
    confidence = st.slider(
        f"Confidence Threshold for {target_fruit.title()}",
        min_value=0.5,
        max_value=0.95,
        value=0.6,
        step=0.05,
        help="Higher values = more strict detection"
    )
    return confidence

def create_input_method_selector():
    st.markdown("### 📷 Choose Input Method")
    
    methods = {
        "📤 Upload Image": "upload",
        "📸 Camera Capture": "camera", 
        "📹 Live Camera": "live"
    }
    
    cols = st.columns(3)
    selected_method = None
    
    for i, (label, method) in enumerate(methods.items()):
        with cols[i]:
            if st.button(label, key=f"method_{method}"):
                selected_method = method
    
    return selected_method or st.session_state.get('input_method', 'upload')

def create_results_display(results, target_fruit):
    if results['has_target_fruit']:
        st.markdown(f"""
        <div class="status-success">
            🎯 {target_fruit.upper()} DETECTED! 
            <br>Confidence: {results['confidence']:.1%}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="status-warning">
            ❌ No {target_fruit.upper()} detected
            <br>Found: {results['predicted_fruit'].title()} ({results['confidence']:.1%})
        </div>
        """, unsafe_allow_html=True)
    
    # Fruit distribution
    st.markdown("### 🍎 Detected Fruits")
    
    fruit_emojis = {
        'apple': '🍎', 'avocado': '🥑', 'banana': '🍌',
        'cherry': '🍒', 'kiwi': '🥝', 'mango': '🥭',
        'orange': '🍊', 'pineapple': '🍍', 'strawberry': '🍓',
        'watermelon': '🍉'
    }
    
    detected_fruits = {k: v for k, v in results['fruit_counts'].items() if v > 0}
    
    if detected_fruits:
        cols = st.columns(min(len(detected_fruits), 4))
        for i, (fruit, count) in enumerate(detected_fruits.items()):
            with cols[i % 4]:
                emoji = fruit_emojis.get(fruit, '🍇')
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 2rem;">{emoji}</div>
                    <h3>{count}</h3>
                    <p>{fruit.title()}</p>
                </div>
                """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="AI Fruit Detector",
        page_icon="🍎",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Create modern header
    create_modern_header()
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("## 🎛️ Configuration Panel")
        
        target_fruit = create_fruit_selector()
        confidence_threshold = create_confidence_slider(target_fruit)
        
        st.markdown("---")
        st.markdown("### 📊 System Info")
        st.info(f"🎯 Target: {target_fruit.title()}")
        st.info(f"⚙️ Confidence: {confidence_threshold:.0%}")
        
        # Load detector
        detector = load_detector(target_fruit)
    
    # Main content area
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("## 📷 Image Input")
        
        # Input method selection
        input_method = st.selectbox(
            "Select input method:",
            ["📤 Upload Image", "📸 Camera Capture", "📹 Live Camera"],
            key="input_selector"
        )
        
        if "Upload" in input_method:
            st.markdown("### Upload Your Image")
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=["jpg", "jpeg", "png"],
                help="Upload a clear image of fruits for best results"
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                if st.button("🔍 Analyze Image", type="primary"):
                    with st.spinner("🤖 AI is analyzing your image..."):
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                        
                        results = detector.detect_fruits_in_basket(image, confidence_threshold)
                        st.session_state.results = results
                        st.session_state.analyzed = True
                        st.success("✅ Analysis complete!")
        
        elif "Camera" in input_method:
            st.markdown("### 📸 Camera Capture")
            st.markdown('<div class="camera-container">', unsafe_allow_html=True)
            
            camera_photo = st.camera_input("Take a photo of your fruits")
            
            if camera_photo:
                image = Image.open(camera_photo).convert('RGB')
                st.image(image, caption="Captured Image", use_container_width=True)
                
                if st.button("🔍 Analyze Photo", type="primary"):
                    with st.spinner("🤖 Processing your photo..."):
                        results = detector.detect_fruits_in_basket(image, confidence_threshold)
                        st.session_state.results = results
                        st.session_state.analyzed = True
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("## 📊 Detection Results")
        
        if hasattr(st.session_state, 'analyzed') and st.session_state.analyzed:
            results = st.session_state.results
            create_results_display(results, target_fruit)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; color: #666;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🤖</div>
                <h3>Ready to Analyze!</h3>
                <p>Upload an image or take a photo to start fruit detection</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer with instructions
    st.markdown("---")
    with st.expander("ℹ️ How to Use This App", expanded=False):
        st.markdown("""
        ### 🚀 Quick Start Guide
        
        1. **Select Target Fruit** 🎯 - Choose which fruit you want to detect
        2. **Adjust Confidence** ⚙️ - Set detection sensitivity (60% recommended)
        3. **Upload Image** 📤 - Choose a clear photo of fruits
        4. **Get Results** 📊 - View detection results and confidence scores
        
        ### 🎯 Supported Fruits
        🍎 Apple • 🥑 Avocado • 🍌 Banana • 🍒 Cherry • 🥝 Kiwi
        🥭 Mango • 🍊 Orange • 🍍 Pineapple • 🍓 Strawberry • 🍉 Watermelon
        
        ### 💡 Tips for Best Results
        - Use well-lit, clear images
        - Ensure fruits are clearly visible
        - Avoid blurry or dark photos
        - Single fruits work better than crowded baskets
        """)

if __name__ == "__main__":
    main()
