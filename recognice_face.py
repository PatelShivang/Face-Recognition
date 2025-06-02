import cv2
import numpy as np
import face_recognition
import sqlite3
import json

# Function to check if face exists in database
def face_exists(encoding):
    conn = sqlite3.connect("face_database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name, encoding FROM faces")
    known_faces = cursor.fetchall()
    conn.close()

    for name, encoding_str in known_faces:
        stored_encoding = np.array(json.loads(encoding_str))
        match = face_recognition.compare_faces([stored_encoding], encoding, tolerance=0.6)
        if match[0]:  # Face matches an existing one
            return name  

    return None  # Face is new

# Function to store new face
def store_face(name, encoding):
    encoding_str = json.dumps(encoding.tolist())  # Convert encoding to JSON string
    conn = sqlite3.connect("face_database.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO faces (name, encoding) VALUES (?, ?)", (name, encoding_str))
    conn.commit()
    conn.close()
    print(f" {name} added to database!")

# Main function to detect and recognize faces
def recognize_and_add_faces():
    cap = cv2.VideoCapture(0)  # Open camera

    # Load known faces from the database
    conn = sqlite3.connect("face_database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name, encoding FROM faces")
    known_faces = cursor.fetchall()
    conn.close()

    known_names = []
    known_encodings = []

    # Convert stored encodings from JSON to numpy arrays
    for name, encoding_str in known_faces:
        known_names.append(name)
        known_encodings.append(np.array(json.loads(encoding_str)))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = face_exists(face_encoding)

            if name:  # Face exists in database
                label = name
            else:  # New face detected
                label = "New Face"
                print(" New face detected! Enter name: ")
                new_name = input("Enter name: ").strip()
                store_face(new_name, face_encoding)
                known_names.append(new_name)
                known_encodings.append(face_encoding)
                label = new_name

            # Draw a rectangle and label the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Face Recognition & Auto Registration", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the recognition system with auto face addition
recognize_and_add_faces()
