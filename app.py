import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
import time

# Try to import MTCNN (primary face detector)
try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print("Warning: facenet-pytorch not installed. Using Haar Cascade only.")
    print("Install with: pip install facenet-pytorch")

# Import model and config
from src.models.emotion_resnet import EmotionResNet
from config import EMOTION_LABELS, IMAGE_SIZE, DEVICE, MODELS_DIR

# Global variables
model = None
mtcnn = None

# Emotion colors for visualization (BGR for OpenCV)
EMOTION_COLORS = {
    "Surprise": (255, 255, 0),    # Cyan (BGR)
    "Fear": (180, 0, 180),        # Purple (BGR)
    "Disgust": (0, 180, 0),       # Green (BGR)
    "Happiness": (0, 220, 255),   # Yellow (BGR)
    "Sadness": (255, 100, 0),     # Blue (BGR)
    "Anger": (0, 0, 255),         # Red (BGR)
    "Neutral": (128, 128, 128)    # Gray (BGR)
}

# Preprocessing transform (same as test transforms)
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model():
    """Load the trained emotion detection model and initialize MTCNN."""
    global model, mtcnn

    model_path = MODELS_DIR / "best_resnet_rafdb.pth"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Initialize emotion model
    model = EmotionResNet(num_classes=7, dropout_rate=0.5, pretrained=False)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(DEVICE)
    model.eval()

    print(f"[Real-Time] Model loaded from {model_path}")
    print(f"[Real-Time] Using device: {DEVICE}")

    # Initialize MTCNN face detector
    if MTCNN_AVAILABLE:
        mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=False,
            device=DEVICE,
            keep_all=True
        )
        print("[Real-Time] MTCNN face detector initialized")
    else:
        print("[Real-Time] Using Haar Cascade face detector (MTCNN not available)")

    return model


def detect_faces_mtcnn(image):
    """
    Detect faces using MTCNN (more accurate).
    Returns list of (x, y, w, h) tuples.
    """
    global mtcnn

    # Convert BGR to RGB for MTCNN
    img_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    boxes, probs, landmarks = mtcnn.detect(img_array, landmarks=True)

    faces = []
    if boxes is not None:
        for i, box in enumerate(boxes):
            if probs[i] > 0.9:  # Only keep high confidence detections
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                faces.append((int(x1), int(y1), int(w), int(h)))

    return faces


def detect_faces_haar(image):
    """
    Detect faces using OpenCV Haar Cascade (fallback).
    Returns list of (x, y, w, h) tuples.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Histogram equalization for better detection
    gray = cv2.equalizeHist(gray)

    # Load Haar Cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def detect_faces(image):
    """
    Detect faces using MTCNN (primary) or Haar Cascade (fallback).
    Returns list of (x, y, w, h) tuples.
    """
    if MTCNN_AVAILABLE and mtcnn is not None:
        faces = detect_faces_mtcnn(image)
        # Fall back to Haar if MTCNN finds nothing
        if len(faces) == 0:
            faces = detect_faces_haar(image)
        return faces
    else:
        return detect_faces_haar(image)


def predict_emotion(face_image):
    """
    Predict emotion for a face image.
    Returns (emotion_label, confidence, all_probabilities).
    """
    global model

    # Convert BGR to RGB
    face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_image_rgb)

    # Ensure RGB
    if face_pil.mode != 'RGB':
        face_pil = face_pil.convert('RGB')

    # Preprocess
    input_tensor = preprocess(face_pil).unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    # Get results
    predicted_class = predicted.item()
    confidence_score = confidence.item()
    all_probs = probabilities[0].cpu().numpy()

    emotion_label = EMOTION_LABELS[predicted_class]

    # Create probability dict
    prob_dict = {EMOTION_LABELS[i]: float(all_probs[i]) for i in range(len(EMOTION_LABELS))}

    return emotion_label, confidence_score, prob_dict


def process_frame(frame):
    """
    Process a video frame: detect faces and predict emotions.
    Returns annotated frame and results.
    """
    results = []
    
    # Detect faces
    faces = detect_faces(frame)

    # Process each face
    for i, (x, y, w, h) in enumerate(faces):
        # Add margin
        margin = int(0.1 * min(w, h))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)

        # Extract face
        face_region = frame[y1:y2, x1:x2]
        
        # Skip if face region is too small
        if face_region.shape[0] < 10 or face_region.shape[1] < 10:
            continue

        # Predict emotion
        emotion, confidence, probabilities = predict_emotion(face_region)

        results.append({
            'face_id': i + 1,
            'bbox': [int(x), int(y), int(w), int(h)],
            'emotion': emotion,
            'confidence': round(confidence * 100, 2)
        })

        # Draw on frame
        color = EMOTION_COLORS.get(emotion, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Add label
        label = f"{emotion} ({confidence * 100:.1f}%)"
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        # Background for text
        cv2.rectangle(
            frame,
            (x, y - text_height - 10),
            (x + text_width + 10, y),
            color,
            -1
        )

        # Text
        cv2.putText(
            frame,
            label,
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness
        )

    return frame, results


def real_time_emotion_detection():
    """Main function for real-time emotion detection from webcam."""
    # Load model
    load_model()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Get webcam properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n" + "="*50)
    print("Real-Time Face Emotion Detection")
    print("="*50)
    print(f"Webcam Resolution: {width}x{height}")
    print(f"Device: {DEVICE}")
    print(f"Face Detector: {'MTCNN' if (MTCNN_AVAILABLE and mtcnn is not None) else 'Haar Cascade'}")
    print(f"Emotions: {list(EMOTION_LABELS.values())}")
    print("="*50)
    print("Press 'q' to quit")
    print("Press 's' to save current frame")
    print("="*50 + "\n")
    
    # Variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    
    # Create window
    cv2.namedWindow('Face Emotion Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Face Emotion Detection', 800, 600)
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Process frame
        processed_frame, results = process_frame(frame)
        
        # Calculate and display FPS
        frame_count += 1
        if frame_count % 10 == 0:
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()
        
        # Display FPS
        cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display number of faces detected
        cv2.putText(processed_frame, f"Faces: {len(results)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display instructions
        cv2.putText(processed_frame, "Press 'q' to quit | 's' to save", (10, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show the frame
        cv2.imshow('Face Emotion Detection', processed_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            # Save current frame
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"emotion_detection_{timestamp}.jpg"
            cv2.imwrite(filename, processed_frame)
            print(f"Frame saved as {filename}")
        elif key == ord('f'):
            # Toggle fullscreen
            cv2.setWindowProperty('Face Emotion Detection', cv2.WND_PROP_FULLSCREEN, 
                                 cv2.WINDOW_FULLSCREEN)
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Real-time detection stopped.")


if __name__ == '__main__':
    real_time_emotion_detection()