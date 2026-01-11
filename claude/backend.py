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

def create_progressive_sketches(image: np.ndarray) -> dict:
    """Generate progressive sketch versions"""
    # Step 1: Basic composition lines
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    composition = cv2.Canny(gray, 100, 200)
    composition = cv2.dilate(composition, np.ones((2, 2), np.uint8), iterations=1)
    
    # Step 2: Basic shapes
    shapes = create_basic_shapes_overlay(image)
    
    # Step 3: Detailed outlines
    detailed = process_image_edges(image)
    
    # Step 4: Mid-tone details
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    mid_tones = cv2.Canny(bilateral, 50, 150)
    
    # Step 5: Final details
    fine_details = cv2.Canny(gray, 30, 100)
    
    return {
        "step_1_composition": encode_image_to_base64(composition),
        "step_2_basic_shapes": encode_image_to_base64(shapes),
        "step_3_outlines": encode_image_to_base64(detailed),
        "step_4_midtones": encode_image_to_base64(mid_tones),
        "step_5_details": encode_image_to_base64(fine_details)
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
        
        # Generate progressive sketch versions
        processed_images = create_progressive_sketches(image)
        
        # Analyze with Claude Vision
        analysis = await analyze_image_with_claude(original_base64)
        
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