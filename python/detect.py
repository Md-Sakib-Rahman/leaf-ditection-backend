import json
import sys
import os

# Get image path from Node.js
image_path = sys.argv[1]

# Verify file exists
if not os.path.exists(image_path):
    print(json.dumps({
        "success": False,
        "error": "Image file not found"
    }))
    sys.exit(1)

# Temporary fake prediction
result = {
    "success": True,
    "filename": os.path.basename(image_path),
    "disease": "Healthy",
    "confidence": 0.99
}

# Return JSON to Node.js
print(json.dumps(result))