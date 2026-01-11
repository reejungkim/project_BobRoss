"""
AI Sketching Tutor - Streamlit Frontend
Quick MVP for testing the sketching tutorial generation
"""

import streamlit as st
import requests
import base64
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="AI Sketching Tutor",
    page_icon="🎨",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E86AB;
        margin-bottom: 1rem;
    }
    .step-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #2E86AB;
    }
    .step-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2E86AB;
        margin-bottom: 0.5rem;
    }
    .instruction-text {
        font-size: 1.1rem;
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

# Backend API URL
API_URL = "http://localhost:8000"

def call_api(image_bytes):
    """Call the FastAPI backend"""
    try:
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
        response = requests.post(f"{API_URL}/api/analyze-sketch", files=files)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to backend. Make sure the FastAPI server is running on port 8000.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def display_base64_image(base64_str, caption=""):
    """Display base64 encoded image"""
    try:
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, caption=caption, use_column_width=True)
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")

# Header
st.markdown('<div class="main-header">🎨 AI Sketching Tutor</div>', unsafe_allow_html=True)
st.markdown("### Transform any photo into a step-by-step drawing guide")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📖 How It Works")
    st.write("""
    1. **Upload** a photo you want to learn to draw
    2. **AI Analyzes** the image and breaks it down
    3. **Follow** the 5-step progressive tutorial
    4. **Practice** with simplified reference images
    """)
    
    st.header("✨ Features")
    st.write("""
    - Geometric shape decomposition
    - Progressive detail building
    - Edge detection overlays
    - Beginner-friendly instructions
    """)
    
    st.header("🎯 Best Results")
    st.write("""
    - Use clear, well-lit photos
    - Simple subjects work best initially
    - Ensure good contrast
    - Avoid overly complex scenes
    """)

# Main content
uploaded_file = st.file_uploader(
    "Upload an image to get started",
    type=["jpg", "jpeg", "png"],
    help="Choose a photo you'd like to learn to draw"
)

if uploaded_file is not None:
    # Display original image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Original Image")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("🔍 Quick Preview")
        st.info("Your image will be analyzed to create a step-by-step drawing guide tailored for beginners.")
    
    # Analyze button
    if st.button("🎨 Generate Drawing Tutorial", type="primary"):
        with st.spinner("🤖 AI is analyzing your image and creating a personalized tutorial..."):
            # Convert image to bytes
            img_byte_arr = io.BytesIO()
            
            # Convert RGBA to RGB if needed (for PNG with transparency)
            if image.mode in ('RGBA', 'LA', 'P'):
                # Create a white background
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                # Paste the image on white background
                background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            # Call API
            result = call_api(img_byte_arr)
            
            if result:
                st.success("✅ Tutorial Generated Successfully!")
                
                # Store in session state
                st.session_state['tutorial'] = result

# Display tutorial if available
if 'tutorial' in st.session_state:
    tutorial = st.session_state['tutorial']
    
    st.markdown("---")
    st.header("📚 Your Personalized Drawing Tutorial")
    
    # Tutorial metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Difficulty Level", tutorial['difficulty_level'].title())
    with col2:
        st.metric("Estimated Time", tutorial['estimated_time'])
    with col3:
        st.metric("Total Steps", tutorial['total_steps'])
    
    st.markdown("---")
    
    # Create tabs for each step
    tab_names = [f"Step {i+1}" for i in range(len(tutorial['steps']))]
    tab_names.append("All Reference Images")
    tabs = st.tabs(tab_names)
    
    # Display each step in its tab
    for idx, step in enumerate(tutorial['steps']):
        with tabs[idx]:
            st.markdown(f'<div class="step-title">Step {step["step_number"]}: {step["title"]}</div>', 
                       unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### 📝 Instructions")
                st.markdown(f'<div class="instruction-text">{step["description"]}</div>', 
                           unsafe_allow_html=True)
                st.markdown("---")
                st.markdown(f'<div class="instruction-text">{step["instructions"]}</div>', 
                           unsafe_allow_html=True)
                
                st.markdown("#### 🔷 Key Shapes")
                for shape in step["key_shapes"]:
                    st.markdown(f"- {shape}")
                
                st.markdown("#### 🎯 Focus Areas")
                for area in step["focus_areas"]:
                    st.markdown(f"- {area}")
            
            with col2:
                st.markdown("#### 🖼️ Reference Image")
                # Display corresponding processed image
                image_key = f"step_{step['step_number']}_" + [
                    "composition", "basic_shapes", "outlines", "midtones", "details"
                ][idx]
                
                if image_key in tutorial['processed_images']:
                    display_base64_image(
                        tutorial['processed_images'][image_key],
                        f"Step {step['step_number']} Reference"
                    )
    
    # All reference images tab
    with tabs[-1]:
        st.header("🖼️ All Reference Images")
        st.write("View all processed reference images side by side")
        
        cols = st.columns(3)
        image_names = [
            ("step_1_composition", "Step 1: Composition"),
            ("step_2_basic_shapes", "Step 2: Basic Shapes"),
            ("step_3_outlines", "Step 3: Outlines"),
            ("step_4_midtones", "Step 4: Mid-tones"),
            ("step_5_details", "Step 5: Details")
        ]
        
        for idx, (key, name) in enumerate(image_names):
            with cols[idx % 3]:
                if key in tutorial['processed_images']:
                    display_base64_image(tutorial['processed_images'][key], name)
    
    # # Download section
    # st.markdown("---")
    # st.header("💾 Save Your Tutorial")
    
    # if st.button("📥 Download Tutorial as JSON"):
    #     import json
    #     tutorial_json = json.dumps(tutorial, indent=2)
    #     st.download_button(
    #         label="Download JSON",
    #         data=tutorial_json,
    #         file_name="drawing_tutorial.json",
    #         mime="application/json"
    #     )

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888;'>
        <p>Made with ❤️ </p>
    </div>
""", unsafe_allow_html=True)