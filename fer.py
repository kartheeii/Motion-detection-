import cv2
import numpy as np
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import glob

# Function to collect data and save frames with motion
def collect_motion_data(camera_index=0, save_path='motion_data'):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Camera not found.")
        return
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    prev_frame = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is None:
            prev_frame = gray_frame
            continue
        
        diff = cv2.absdiff(prev_frame, gray_frame)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            

            # Save frames where motion is detected
            frame_count += 1
            cv2.imwrite(os.path.join(save_path, f"frame_{frame_count}.jpg"), frame)
        
        prev_frame = gray_frame
        
        # Display the annotated frame
        cv2.imshow('Motion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Function to create and train a neural network model
def train_motion_model(data_path='motion_data', epochs=10):
    # Load data
    images = []
    labels = []
    for image_path in glob.glob(os.path.join(data_path, '*.jpg')):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (64, 64))  # Resize for consistency
        images.append(img)
        labels.append(1)  # Assuming all collected frames are with motion

    images = np.array(images, dtype='float32') / 255.0  # Normalize pixel values
    labels = to_categorical(labels, num_classes=2)
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    # Define a simple CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax')
    ])  
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
    
    # Save the model
    model.save('camera_model.h5')

# Function to detect motion using both traditional and model-based methods
def detect_motion_with_combined_approach(camera_index=0, model_path='camera_model.h5'):
    # Load the trained model
    model = load_model(model_path)
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Camera not found.")
        return
    
    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is None:
            prev_frame = gray_frame
            continue
        
        # Traditional motion detection
        diff = cv2.absdiff(prev_frame, gray_frame)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) < 500:  # Filter out small contours
                continue
            
            # Draw bounding box around the moving regions
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Extract the region of interest (ROI) for neural network prediction
            roi = frame[y:y + h, x:x + w]
            if roi.size > 0:  # Ensure ROI is valid
                roi_resized = cv2.resize(roi, (64, 64))
                input_frame = np.expand_dims(roi_resized, axis=0) / 255.0
                
                # Predict using the model
                prediction = model.predict(input_frame)
                motion_detected = np.argmax(prediction[0]) == 1
                
                # Annotate the frame based on model prediction
                if motion_detected:
                    cv2.putText(frame, "Movement", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        prev_frame = gray_frame
        
        # Display the annotated frame
        cv2.imshow('Combined Motion Detection', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Collect data for training
    collect_motion_data()
    
    # Train the model with collected data
    train_motion_model()

    # Use the combined approach for real-time motion detection
    detect_motion_with_combined_approach()
