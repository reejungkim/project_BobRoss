# 🎨 AI Sketching Tutor (Bob Ross Project)

An AI-powered application that transforms any photo into a step-by-step drawing tutorial. Upload an image and receive a personalized 5-step progressive drawing guide with detailed instructions, key shapes, and reference images.

## ✨ Features

- **AI-Powered Analysis**: Uses Claude AI to analyze images and break them down into learnable steps
- **Progressive Tutorials**: 5-step progressive building approach (composition → basic shapes → outlines → midtones → details)
- **Smart Image Processing**: Generates simplified reference images for each step using OpenCV
- **Detailed Instructions**: Each step includes:
  - Title and description
  - Detailed instructions
  - Key geometric shapes to use
  - Focus areas to pay attention to
- **Difficulty Assessment**: Automatically determines difficulty level (beginner/intermediate/advanced)
- **Time Estimation**: Provides estimated completion time for the tutorial
- **Interactive Web Interface**: Clean, user-friendly Streamlit frontend

## 🏗️ Architecture

The project consists of two main components:

### Backend (`claude/backend.py`)
- **FastAPI** REST API server
- **Claude AI** integration for image analysis
- **OpenCV** for progressive sketch generation
- Generates 5 progressive sketch versions:
  1. Composition & Basic Shapes
  2. Refined Basic Shapes
  3. Detailed Outlines
  4. Mid-tones & Texture Indicators
  5. Complete Sketch with Full Shading

### Frontend (`claude/frontend.py`)
- **Streamlit** web application
- Image upload interface
- Tutorial display with tabs for each step
- Reference image gallery
- Responsive layout with custom styling

## 📋 Prerequisites

- Python 3.8+
- Anthropic API key (for Claude AI)

## 🚀 Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd 02_project_BobRoss
   ```

2. **Install dependencies**
   ```bash
   cd claude
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   ANTHROPIC_API_KEY=your_api_key_here
   ```

4. **Start the backend server**
   ```bash
   cd claude
   python backend.py
   ```
   The API will be available at `http://localhost:8000`

5. **Start the frontend** (in a new terminal)
   ```bash
   cd claude
   streamlit run frontend.py
   ```
   The web interface will open at `http://localhost:8501`

## 📡 API Endpoints

### `POST /api/analyze-sketch`
Upload an image and receive a step-by-step drawing tutorial.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Image file (jpg, jpeg, png)

**Response:**
```json
{
  "original_image": "base64_encoded_string",
  "total_steps": 5,
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
  ],
  "processed_images": {
    "step_1_composition": "base64_encoded_string",
    "step_2_basic_shapes": "base64_encoded_string",
    "step_3_outlines": "base64_encoded_string",
    "step_4_midtones": "base64_encoded_string",
    "step_5_details": "base64_encoded_string"
  }
}
```

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "AI Sketching Tutor v2"
}
```

### `GET /`
Root endpoint with API information.

## 🎯 Usage

1. **Open the Streamlit app** in your browser (usually `http://localhost:8501`)

2. **Upload an image** using the file uploader
   - Supported formats: JPG, JPEG, PNG
   - Best results with clear, well-lit photos
   - Simple subjects work best initially

3. **Click "Generate Drawing Tutorial"** to analyze the image

4. **Follow the tutorial**:
   - Review the difficulty level and estimated time
   - Navigate through each step using the tabs
   - Read the detailed instructions for each step
   - Use the reference images as visual guides
   - View all reference images in the "All Reference Images" tab

## 📦 Dependencies

### Backend
- `fastapi>=0.109.0` - Web framework
- `uvicorn[standard]>=0.27.0` - ASGI server
- `anthropic>=0.18.0` - Claude AI client
- `opencv-python>=4.8.0` - Image processing
- `Pillow>=10.0.0` - Image manipulation
- `numpy>=1.24.0,<2.0.0` - Numerical operations
- `python-dotenv` - Environment variable management

### Frontend
- `streamlit>=1.30.0` - Web interface
- `requests>=2.31.0` - HTTP client

## 🔧 Configuration

The frontend can be configured to use a different backend URL by modifying the `API_URL` variable in `frontend.py`:

```python
# Local development
API_URL = "http://localhost:8000"

# Production (e.g., Render)
API_URL = "https://sketch-tutor-backend.onrender.com"
```

## 🎨 How It Works

1. **Image Upload**: User uploads an image through the Streamlit interface
2. **AI Analysis**: The image is sent to Claude AI, which analyzes it and breaks it down into 5 sequential steps
3. **Progressive Sketch Generation**: OpenCV processes the image to create 5 progressive versions:
   - Step 1: Basic geometric shapes and composition lines
   - Step 2: Refined shapes with more detail
   - Step 3: Detailed outlines
   - Step 4: Mid-tones and texture indicators
   - Step 5: Complete sketch with full shading and cross-hatching
4. **Tutorial Display**: The frontend displays the tutorial with instructions, key shapes, focus areas, and reference images for each step

## 🐛 Troubleshooting

- **Connection Error**: Make sure the backend server is running on port 8000
- **API Key Error**: Verify your `ANTHROPIC_API_KEY` is set in the `.env` file
- **Image Processing Error**: Ensure the uploaded image is in a supported format (JPG, JPEG, PNG)

## 📝 Notes

- The backend uses Claude Sonnet 4 (`claude-sonnet-4-20250514`) for image analysis
- Images are processed using OpenCV with various edge detection and contour finding techniques
- The progressive sketch generation uses geometric approximation and hatching techniques to create simplified reference images

## 📄 License

[Add your license here]

## 👤 Author

[Add your name/info here]
