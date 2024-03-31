import cv2
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model(r'C:\Users\Ardhaoui Amine\Desktop\AI NIGHT CHALLENGE\glass_bangle_defect_detection_cnn_model.h5')

# Define constants
IMAGE_WIDTH, IMAGE_HEIGHT = 150, 150

# Dictionary to map class indices to class labels
class_labels = {0: 'good', 1: 'broken', 2: 'defect'}

# Function to preprocess frame
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
    normalized_frame = resized_frame / 255.0
    return normalized_frame.reshape(1, IMAGE_WIDTH, IMAGE_HEIGHT, 3)

# Function to start object detection
def start_detection():
    # Open camera feed
    cap = cv2.VideoCapture(0)

    def detect_and_display():
        nonlocal cap
        ret, frame = cap.read()
        if not ret:
            return

        # Preprocess the frame
        preprocessed_frame = preprocess_frame(frame)

        # Predict the class
        predictions = model.predict(preprocessed_frame)
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = class_labels[predicted_class_index]

        # Overlay predicted class label on frame
        cv2.putText(frame, predicted_class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw bounding box around glass bangle (for illustration purposes)
        if predicted_class_label != 'good':  # Draw bounding box only if not classified as 'good'
            # Adjust bounding box coordinates based on the model's prediction
            # Replace the following example coordinates with dynamically calculated ones
            cv2.rectangle(frame, (50, 50), (100, 100), (255, 0, 0), 2)  

        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to ImageTk format
        img_tk = ImageTk.PhotoImage(Image.fromarray(frame_rgb))

        # Update the label with the new frame
        lbl_camera.img_tk = img_tk
        lbl_camera.config(image=img_tk)

        # Call detect_and_display again after 1ms
        root.after(1, detect_and_display)

    detect_and_display()

# Function to stop object detection
def stop_detection():
    cv2.destroyAllWindows()

# Function to reset the camera feed
def reset_camera():
    pass  # Placeholder function for reset functionality

# Function to exit the application
def exit_application():
    stop_detection()  # Stop object detection if running
    root.quit()  # Quit Tkinter application

# Create Tkinter window
root = Tk()
root.title("Object Detection")

# Set window to full screen
root.attributes('-fullscreen', True)

# Create a label to display camera feed
lbl_camera = Label(root)
lbl_camera.pack()

# Create button to start object detection
btn_start = Button(root, text="Start Detection", command=start_detection, bg="green", fg="white")
btn_start.pack()

# Create button to stop object detection
btn_stop = Button(root, text="Stop Detection", command=stop_detection, bg="red", fg="white")
btn_stop.pack()

# Create button to reset camera feed
btn_reset = Button(root, text="Reset Camera", command=reset_camera, bg="blue", fg="white")
btn_reset.pack()

# Create button to exit application
btn_exit = Button(root, text="Exit", command=exit_application, bg="black", fg="white")
btn_exit.pack()

# Run Tkinter main loop
root.mainloop()
