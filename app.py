from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
import numpy as np
import base64
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

@app.route('/encode_face', methods=['POST'])
def encode_face():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # Decode base64 image
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB (face_recognition expects RGB)
        image = image.convert('RGB')
        image_np = np.array(image)

        # Detect and encode
        # face_encodings returns a list of 128-dimensional face encodings
        face_encodings = face_recognition.face_encodings(image_np)
        
        if len(face_encodings) > 0:
            # Return the first face found converted to a list
            return jsonify({'encoding': face_encodings[0].tolist()})
        
        return jsonify({'encoding': None})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)