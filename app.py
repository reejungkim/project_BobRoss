import streamlit as st
import cv2
import numpy as np
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import base64
import io
import os 
from dotenv import load_dotenv

# --- 1. CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="Cilantrosso Sketch Tutor", layout="wide")

st.title("🎨 Bob Ross AI Sketch Tutor")
st.markdown("""
    *“We don't make mistakes, just happy little accidents.”*
    Upload a photo, and I'll help you break it down into a beautiful sketch.
""")

# # Sidebar for API Key (Keeping it flexible like your speaking-ai app)
# with st.sidebar:
#     openai_api_key = st.text_input("OpenAI API Key", type="password")
#     st.info("This app uses GPT-4o to analyze your photo and guide you.")
load_dotenv('/Users/reejungkim/Documents/00_Git/00_working-in-progress/.env')
openai_api_key = os.getenv("openai")

# --- 2. IMAGE PROCESSING FUNCTIONS ---
def generate_sketch(image):
    # Convert PIL to OpenCV format
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Pre-processing: Grayscale and Blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge Detection (The "Simplified" view)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Invert to get black lines on white background
    sketch = cv2.bitwise_not(edges)
    return sketch

def get_bob_ross_advice(image, api_key):
    if not api_key:
        return "Please provide an API key to hear from the coach!"
    
    # Convert PIL Image to Base64 for LangChain
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    llm = ChatOpenAI(model="gpt-4o", api_key=api_key)
    
    prompt = [
        HumanMessage(
            content=[
                {"type": "text", "text": "Act as Bob Ross. Look at this photo. Give me one sentence of warm encouragement and one specific tip on which basic shape (circle, line, triangle) to draw first to capture this subject easily."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        )
    ]
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"The coach is taking a nap (Error: {str(e)})"

# --- 3. APP LOGIC ---
uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Original and Sketch side-by-side
    col1, col2 = st.columns(2)
    
    original_image = Image.open(uploaded_file)
    
    with col1:
        st.subheader("Your Photo")
        st.image(original_image, use_column_width=True)
        
    with col2:
        st.subheader("The Sketch Guide")
        sketch_img = generate_sketch(original_image)
        st.image(sketch_img, use_column_width=True)

    # --- 4. AI COACHING SECTION ---
    st.divider()
    st.subheader("🖌️ Coach's Guidance")
    
    if st.button("Get Bob Ross's Advice"):
        with st.spinner("The AI coach is looking at your happy little photo..."):
            advice = get_bob_ross_advice(original_image, openai_api_key)
            st.chat_message("assistant", avatar="🎨").write(advice)
            st.balloons()

    # --- 5. HOW TO USE THIS ---
    with st.expander("How to draw with this?"):
        st.write("""
            1. Place a piece of paper on your desk.
            2. Lean your phone or tablet against a stand.
            3. Look at the **Sketch Guide** and try to find the 'big' shapes first.
            4. Don't worry about the details yet—just follow the happy little lines!
        """)

else:
    st.info("Please upload an image to start your drawing journey.")