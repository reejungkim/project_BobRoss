"""
AI-Powered Sketching Tutor Backend v2
Generates actual instructional drawing guides for each step
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import anthropic
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Sketching Tutor v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Anthropic client
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not set!")

print(f"✅ API Key loaded: {api_key[:20]}...")
client = anthropic.Anthropic(api_key=api_key)

# Response Models
class DrawingStep(BaseModel):
    step_number: int
    title: str
    description: str
    instructions: str
    key_shapes: List[str]
    focus_areas: List[str]

class SketchingGuide(BaseModel):
    original_image: str
    total_steps: int
    difficulty_level: str
    estimated_time: str
    steps: List[DrawingStep]
    processed_images: dict

def encode_image_to_base64(image: np.ndarray) -> str:
    """Convert numpy array to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def pil_to_cv2(pil_image):
    """Convert PIL Image to OpenCV format"""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    """Convert OpenCV image to PIL format"""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

async def generate_step_image(image_base64: str, step_number: int, step_description: str) -> str:
    """Generate a drawing guide for a specific step using Claude"""
    
    prompts = {
        1: """Create a simple instructional drawing guide showing ONLY the basic geometric shapes and composition lines for this image. 
        
Instructions:
- Draw simple geometric shapes (circles, rectangles, ovals, triangles) that approximate the main objects
- Add composition guideline (center lines, rule of thirds)
- Use simple black lines on white background
- NO detailed edges, just basic shapes
- Think like a drawing teacher showing the first step
        
Draw this as if you're teaching someone to sketch - show only the foundational shapes.""",
        
        2: """Building on basic shapes, show the major forms and proportions.

Instructions:
- Refine the geometric shapes into more accurate outlines
- Show how shapes connect to each other
- Add proportion markers
- Still simplified, but more accurate than step 1
- Use medium-weight lines on white background

This is step 2 - more refined than basic shapes, but still simplified.""",
        
        3: """Show the main outlines and contours with more detail.

Instructions:
- Draw clean, confident outlines of all major forms
- Add important internal contours
- Show where major shadows will fall
- More detailed than step 2, but still clean and simple
- Use darker lines for main contours

This is step 3 - clear outlines that define the subject.""",
        
        4: """Add secondary details and indicate texture/shading areas.

Instructions:
- Keep all outlines from step 3
- Add secondary details (facial features, texture indicators)
- Use lighter lines to indicate where shading will go
- Add hatching marks in shadow areas
- Show texture patterns

This is step 4 - adding details and preparing for shading.""",
        
        5: """Complete drawing with all details and shading.

Instructions:
- Include everything from previous steps
- Add fine details
- Show full shading with cross-hatching
- Add texture throughout
- Create depth with values
- This should look like a finished pencil sketch

This is step 5 - the complete, detailed drawing."""
    }
    
    prompt = prompts.get(step_number, prompts[3])
    
    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": f"{prompt}\n\nStep {step_number} focus: {step_description}\n\nGenerate the instructional drawing guide for this step."
                        }
                    ]
                }
            ]
        )
        
        # Note: Claude cannot generate images directly, so we'll use traditional CV approach
        # but with better logic based on step number
        return None
        
    except Exception as e:
        print(f"Error generating step image: {e}")
        return None

def create_progressive_sketches_smart(image: np.ndarray, analysis: dict) -> dict:
    """Generate progressive sketch versions with intelligence based on step descriptions"""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Extract what each step should focus on from AI analysis
    steps_info = analysis.get("steps", [])
    
    # Step 1: Basic geometric shapes and composition
    step1 = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    # Find major contours for geometric approximation
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    edges = cv2.Canny(blurred, 20, 60)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    for contour in contours:
        # Highly simplified geometric approximation
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) <= 4:
            cv2.polylines(step1, [approx], True, (100, 100, 100), 2)
        else:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius > 10:
                cv2.circle(step1, (int(x), int(y)), int(radius), (100, 100, 100), 2)
    
    # Add composition lines
    cv2.line(step1, (w//2, 0), (w//2, h), (200, 200, 200), 1)
    cv2.line(step1, (0, h//2), (w, h//2), (200, 200, 200), 1)
    cv2.line(step1, (w//3, 0), (w//3, h), (220, 220, 220), 1)
    cv2.line(step1, (2*w//3, 0), (2*w//3, h), (220, 220, 220), 1)
    
    # Step 2: Build on step 1 - add more refined shapes
    step2 = step1.copy()
    edges_2 = cv2.Canny(cv2.GaussianBlur(gray, (9, 9), 0), 30, 90)
    contours_2, _ = cv2.findContours(edges_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_2 = sorted(contours_2, key=cv2.contourArea, reverse=True)[:15]
    
    for contour in contours_2:
        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.polylines(step2, [approx], True, (80, 80, 80), 1)
    
    # Step 3: Add detailed outlines
    step3 = np.ones((h, w, 3), dtype=np.uint8) * 255
    # Start with step 2 as base
    mask2 = cv2.cvtColor(step2, cv2.COLOR_BGR2GRAY)
    step3[mask2 < 250] = step2[mask2 < 250]
    
    # Add more detailed contours
    edges_3 = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    contours_3, _ = cv2.findContours(edges_3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(step3, contours_3, -1, (60, 60, 60), 1)
    
    # Step 4: Add details and texture indicators
    step4 = step3.copy()
    
    # Add finer details
    edges_4 = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 30, 90)
    step4[edges_4 > 0] = [100, 100, 100]
    
    # Add light hatching in darker areas to indicate shading
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            if gray[i, j] < 120:
                cv2.line(step4, (j, i), (j+4, i+4), (160, 160, 160), 1)
    
    # Step 5: Complete sketch with full shading
    step5 = step4.copy()
    
    # Add cross-hatching for darker areas
    for i in range(0, h, 6):
        for j in range(0, w, 6):
            intensity = gray[i, j]
            if intensity < 100:
                # Dark areas: dense cross-hatching
                cv2.line(step5, (j, i), (j+4, i+4), (140, 140, 140), 1)
                cv2.line(step5, (j+4, i), (j, i+4), (140, 140, 140), 1)
            elif intensity < 150:
                # Mid-tones: single hatching
                cv2.line(step5, (j, i), (j+3, i+3), (170, 170, 170), 1)
    
    # Add all fine details
    edges_5 = cv2.Canny(gray, 20, 70)
    step5[edges_5 > 0] = [40, 40, 40]
    
    return {
        "step_1_composition": encode_image_to_base64(step1),
        "step_2_basic_shapes": encode_image_to_base64(step2),
        "step_3_outlines": encode_image_to_base64(step3),
        "step_4_midtones": encode_image_to_base64(step4),
        "step_5_details": encode_image_to_base64(step5)
    }

async def analyze_image_with_claude(image_base64: str):
    """Analyze image with Claude"""
    
    system_prompt = """You are a master drawing instructor. Break down this image into 5 sequential drawing steps.

For each step, provide:
- step_number (1-5)
- title
- description
- instructions (detailed, specific)
- key_shapes (geometric shapes to use)
- focus_areas (what to pay attention to)

Return ONLY valid JSON matching this exact structure:
{
  "difficulty_level": "beginner|intermediate|advanced",
  "estimated_time": "X minutes",
  "steps": [
    {
      "step_number": 1,
      "title": "Composition & Basic Shapes",
      "description": "...",
      "instructions": "...",
      "key_shapes": ["circle", "rectangle"],
      "focus_areas": ["proportions", "placement"]
    }
  ]
}"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": "Analyze this image and create a 5-step drawing tutorial. Return ONLY the JSON, no other text."
                        }
                    ]
                }
            ],
            system=system_prompt
        )
        
        response_text = message.content[0].text
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            analysis = json.loads(json_str)
            return analysis
        else:
            raise ValueError("No valid JSON found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")

@app.post("/api/analyze-sketch", response_model=SketchingGuide)
async def analyze_sketch(file: UploadFile = File(...)):
    """Upload an image and receive a step-by-step drawing tutorial"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        original_base64 = encode_image_to_base64(image)
        
        # Analyze with Claude
        analysis = await analyze_image_with_claude(original_base64)
        
        # Generate smart progressive sketches
        processed_images = create_progressive_sketches_smart(image, analysis)
        
        guide = SketchingGuide(
            original_image=original_base64,
            total_steps=len(analysis["steps"]),
            difficulty_level=analysis.get("difficulty_level", "intermediate"),
            estimated_time=analysis.get("estimated_time", "30-45 minutes"),
            steps=[DrawingStep(**step) for step in analysis["steps"]],
            processed_images=processed_images
        )
        
        return guide
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AI Sketching Tutor v2"}

@app.get("/")
async def root():
    return {
        "message": "AI Sketching Tutor API v2",
        "version": "2.0.0",
        "improvements": "Progressive building: each step adds to the previous"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
