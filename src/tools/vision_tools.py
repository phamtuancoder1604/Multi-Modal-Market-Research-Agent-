import io
import httpx
from PIL import Image
import base64
def multimodal_vision_tool(image_path: str) -> str:
    """
    Accepts an image path of a chart or diagram.
    Uses Pillow to normalize the image and transmits it to a local Ollama 
    Vision LLM to convert the visual layout into descriptive Markdown tables and logic.
    """
    print(f"Processing image asset via Ollama Vision: {image_path}...")
    try:
        # Open and normalize image format using Pillow
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.thumbnail((512,512)) 
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=60)
            img_bytes = buffered.getvalue()

        # Define explicit prompt instructions to convert diagram to structural text
        prompt = (
            "You are a technical document parser. Analyze this abstract block diagram or chart structure.\n"
            "1. List all components, labels, and numbers visible in a standard Markdown table format.\n"
            "2. Trace and describe the connection flow or logical links from left to right."
        )
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        # Execute request against local Ollama generation endpoint
        response = httpx.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llava:7b",
                "prompt": prompt,
                "images": [img_base64],
                "stream": False
            },
            timeout=None
        )
        
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            return f"Ollama Vision API error: Status code {response.status_code}"
            
    except Exception as e:
        return f"Failed to process multimodal image source: {str(e)}"

if __name__ == "__main__":
    test_image = r"C:\Users\ASUS\Project\advanced_research_agent\data\raw\multimodal\diagram.png"
    
    # Create a dummy image file if it does not exist for testing purposes
    import os
    if not os.path.exists(test_image):
        print(f"Test image not found. Creating a blank placeholder at: {test_image}")
        img = Image.new('RGB', (300, 300), color = 'red')
        img.save(test_image)
        
    result = multimodal_vision_tool(test_image)
    print("\nVision Tool Output:")
    print(result)