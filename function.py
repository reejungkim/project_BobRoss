import cv2
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import base64

def process_image_to_sketch(image_path):
    # 1. Load the image
    img = cv2.imread(image_path)
    
    # 2. Convert to Grayscale & Blur to remove noise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Canny Edge Detection (The Stencil)
    # Adjust thresholds to get more/fewer "happy little lines"
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    
    # 4. Invert so it looks like a sketch (black lines on white background)
    sketch = cv2.bitwise_not(edges)
    
    cv2.imwrite('step_1_sketch.png', sketch)
    return 'step_1_sketch.png'

def get_bob_ross_instruction(image_path):
    # Encode image to base64 for the LLM
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    llm = ChatOpenAI(model="gpt-4o", max_tokens=200)
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Act as Bob Ross. Look at this photo and give me one encouraging sentence on how to start sketching the most basic shapes of this object. Keep it under 30 words."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    )
    
    response = llm.invoke([message])
    return response.content

# Execution
# sketch_path = process_image_to_sketch("input_photo.jpg")
# instruction = get_bob_ross_instruction("input_photo.jpg")
# print(f"Coach says: {instruction}")