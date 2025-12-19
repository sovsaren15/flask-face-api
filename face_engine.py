import face_recognition
import numpy as np
from PIL import Image
import io

class FaceEngine:
    """Face recognition engine using face_recognition library"""
    
    def __init__(self, model='hog'):
        """
        Initialize face engine
        model: 'hog' (faster, less accurate) or 'cnn' (slower, more accurate)
        """
        self.model = model
    
    def encode_face(self, image):
        """
        Encode a face from an image
        Returns: numpy array of face encoding or None if no face found
        """
        try:
            # Convert PIL Image to numpy array
            image_np = np.array(image)
            
            # Find faces in the image
            face_locations = face_recognition.face_locations(image_np, model=self.model)
            
            if not face_locations:
                return None
            
            # Encode the first face found
            face_encodings = face_recognition.face_encodings(
                image_np, 
                face_locations
            )
            
            return face_encodings[0] if face_encodings else None
        
        except Exception as e:
            print(f"Error encoding face: {str(e)}")
            return None
    
    def compare_faces(self, known_encoding, test_encoding, tolerance=0.6):
        """
        Compare two face encodings
        Returns: True if faces match, False otherwise
        """
        try:
            if known_encoding is None or test_encoding is None:
                return False
            
            distance = np.linalg.norm(known_encoding - test_encoding)
            return distance < tolerance
        
        except Exception as e:
            print(f"Error comparing faces: {str(e)}")
            return False
    
    def get_face_distance(self, known_encoding, test_encoding):
        """
        Get the distance between two face encodings (lower = more similar)
        """
        try:
            if known_encoding is None or test_encoding is None:
                return float('inf')
            
            return np.linalg.norm(known_encoding - test_encoding)
        
        except Exception as e:
            print(f"Error calculating distance: {str(e)}")
            return float('inf')
