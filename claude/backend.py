"""
AI-Powered Sketching Tutor Backend
Decomposes uploaded photos into sequential drawing steps
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
from PIL import Image
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="AI Sketching Tutor")

# CORS middleware for frontend integration
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
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set! Please set it using: export ANTHROPIC_API_KEY='your-key'")

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
    step_image: str  # Base64 encoded image matching this step's description

class SketchingGuide(BaseModel):
    original_image: str
    total_steps: int
    difficulty_level: str
    estimated_time: str
    steps: List[DrawingStep]
    processed_images: dict

# Image Processing Functions
def encode_image_to_base64(image: np.ndarray) -> str:
    """Convert numpy array to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def process_image_edges(image: np.ndarray) -> np.ndarray:
    """Apply edge detection for simplified sketch view"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def create_basic_shapes_overlay(image: np.ndarray) -> np.ndarray:
    """Create simplified geometric shape overlay"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Canny(blurred, 30, 100)
    
    # Apply morphological operations to simplify
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    simplified = cv2.erode(dilated, kernel, iterations=1)
    
    return simplified

def draw_geometric_shapes(image: np.ndarray, analysis: dict) -> np.ndarray:
    """Draw geometric shapes based on AI analysis"""
    h, w = image.shape[:2]
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    # Find contours to identify major shapes
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    edges = cv2.Canny(blurred, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area and get the largest ones
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    for contour in contours:
        # Approximate the contour to basic shapes
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Draw based on number of vertices
        if len(approx) == 3:
            # Triangle
            cv2.polylines(canvas, [approx], True, (100, 100, 100), 3)
        elif len(approx) == 4:
            # Rectangle/Square
            cv2.polylines(canvas, [approx], True, (100, 100, 100), 3)
        else:
            # Circle/Ellipse for more complex shapes
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius > 10:
                cv2.circle(canvas, (int(x), int(y)), int(radius), (100, 100, 100), 3)
    
    # Add centerlines for composition
    cv2.line(canvas, (w//2, 0), (w//2, h), (200, 200, 200), 1, cv2.LINE_AA)
    cv2.line(canvas, (0, h//2), (w, h//2), (200, 200, 200), 1, cv2.LINE_AA)
    
    return canvas

def create_progressive_sketches(image: np.ndarray, analysis: dict) -> dict:
    """Generate progressive sketch versions based on AI analysis"""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Composition guidelines with basic shapes
    step1 = draw_geometric_shapes(image, analysis)
    
    # Step 2: Major shapes with proportions
    step2 = np.ones((h, w, 3), dtype=np.uint8) * 255
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 150)
    # Simplify edges
    kernel = np.ones((3, 3), np.uint8)
    simplified = cv2.dilate(edges, kernel, iterations=2)
    simplified = cv2.erode(simplified, kernel, iterations=2)
    step2[simplified > 0] = [100, 100, 100]
    
    # Step 3: Refined outlines
    step3 = np.ones((h, w, 3), dtype=np.uint8) * 255
    edges = cv2.Canny(gray, 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(step3, contours, -1, (80, 80, 80), 2)
    
    # Step 4: Add mid-tones and texture hints
    step4 = np.ones((h, w, 3), dtype=np.uint8) * 255
    # Copy step 3
    step4 = step3.copy()
    # Add lighter edges for texture
    fine_edges = cv2.Canny(gray, 20, 80)
    step4[fine_edges > 0] = [150, 150, 150]
    
    # Step 5: Detailed sketch with shading zones
    step5 = np.ones((h, w, 3), dtype=np.uint8) * 255
    # Create shading map
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    edges_detailed = cv2.Canny(enhanced, 30, 100)
    
    # Add hatching for darker areas
    for i in range(0, h, 10):
        for j in range(0, w, 10):
            if gray[i, j] < 100:
                cv2.line(step5, (j, i), (j+5, i+5), (180, 180, 180), 1)
    
    step5[edges_detailed > 0] = [60, 60, 60]
    
    return {
        "step_1_composition": encode_image_to_base64(step1),
        "step_2_basic_shapes": encode_image_to_base64(step2),
        "step_3_outlines": encode_image_to_base64(step3),
        "step_4_midtones": encode_image_to_base64(step4),
        "step_5_details": encode_image_to_base64(step5)
    }

async def analyze_image_with_claude(image_base64: str) -> SketchingGuide:
    """Use Claude Vision to analyze image and generate drawing steps"""
    
    system_prompt = """You are a master drawing instructor with decades of experience teaching beginners to draw from reference photos. Your expertise is in breaking down complex images into simple, achievable steps using basic geometric shapes and progressive detail building.

Analyze the uploaded image and create a 5-step drawing tutorial. Focus on:
1. Overall composition and proportion guidelines
2. Basic geometric shapes that form the foundation
3. Primary outlines and contours
4. Secondary details and textures
5. Final refinements and shading guidance

For each step, provide:
- A clear title
- Detailed description of what to focus on
- Step-by-step instructions
- Key geometric shapes to use (circles, rectangles, triangles, ovals, etc.)
- Specific areas to focus attention on

Return your response as a JSON object matching this structure:
{
  "difficulty_level": "beginner|intermediate|advanced",
  "estimated_time": "X minutes",
  "steps": [
    {
      "step_number": 1,
      "title": "Step title",
      "description": "What this step accomplishes",
      "instructions": "Detailed step-by-step instructions",
      "key_shapes": ["shape1", "shape2"],
      "focus_areas": ["area1", "area2"]
    }
  ]
}"""

    user_prompt = """Analyze this image and create a comprehensive 5-step drawing tutorial for a beginner artist. Break down the subject into simple geometric shapes first, then progressively add detail. Be specific about proportions, positioning, and techniques."""

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
                            "text": user_prompt
                        }
                    ]
                }
            ],
            system=system_prompt
        )
        
        # Extract JSON from response
        response_text = message.content[0].text
        
        # Try to find JSON in the response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            analysis = json.loads(json_str)
            return analysis
        else:
            raise ValueError("No valid JSON found in response")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")

# API Endpoints
@app.post("/api/analyze-sketch", response_model=SketchingGuide)
async def analyze_sketch(file: UploadFile = File(...)):
    """
    Upload an image and receive a step-by-step drawing tutorial
    """
    try:
        # Read and validate image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Encode original image
        original_base64 = encode_image_to_base64(image)
        
        # Analyze with Claude Vision first
        analysis = await analyze_image_with_claude(original_base64)
        
        # Generate progressive sketch versions based on analysis
        processed_images = create_progressive_sketches(image, analysis)
        
        # Construct response
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
    """Health check endpoint"""
    return {"status": "healthy", "service": "AI Sketching Tutor"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Sketching Tutor API",
        "version": "1.0.0",
        "endpoints": {
            "/api/analyze-sketch": "POST - Upload image for analysis",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)