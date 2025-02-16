import cv2
import face_recognition

def load_and_detect_face(image_path):
    """Loads an image and detects faces by drawing rectangles."""
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return None, None

    # Convert image to RGB (OpenCV loads in BGR)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect face locations
    face_locations = face_recognition.face_locations(rgb_image)

    # Draw rectangles around detected faces
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

    # Show the image
    cv2.imshow("Face Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return image, rgb_image  # Return both BGR and RGB versions


def extract_facial_landmarks(image, rgb_image):
    """Extracts and displays facial landmarks on an image."""
    face_landmarks_list = face_recognition.face_landmarks(rgb_image)

    for landmarks in face_landmarks_list:
        for feature, points in landmarks.items():
            for point in points:
                cv2.circle(image, point, 2, (255, 0, 0), -1)  # Blue dots for landmarks

    cv2.imshow("Facial Landmarks", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def deep_face_analysis(image_path):
    """Analyzes the image for facial expressions using DeepFace."""
    from deepface import DeepFace

    try:
        result = DeepFace.analyze(image_path, actions=['emotion'])
        print(result)
    except Exception as e:
        print(f"DeepFace Error: {e}")


def compare_faces(known_path, unknown_path):
    """Compares two faces and determines if they match."""
    # Load known and unknown images
    known_image = face_recognition.load_image_file(known_path)
    unknown_image = face_recognition.load_image_file(unknown_path)

    # Encode faces
    known_encodings = face_recognition.face_encodings(known_image)
    unknown_encodings = face_recognition.face_encodings(unknown_image)

    if not known_encodings or not unknown_encodings:
        print("Error: Could not encode one or both faces.")
        return

    known_encoding = known_encodings[0]
    unknown_encoding = unknown_encodings[0]

    # Compare faces
    results = face_recognition.compare_faces([known_encoding], unknown_encoding)
    print("Match Found" if results[0] else "No Match")


# Example Usage
if __name__ == "__main__":
    image_path = "face.jpg"

    # Detect face and landmarks
    image, rgb_image = load_and_detect_face(image_path)
    if image is not None and rgb_image is not None:
        extract_facial_landmarks(image, rgb_image)

    # Run deep face analysis
    deep_face_analysis(image_path)

    # Compare two faces
    compare_faces("Images/smile.jpeg", "Images/frown.jpeg")