import cv2
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from keras.models import load_model
from keras.utils import img_to_array
import numpy as np

class EmotionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detection App")

        # Load emotion detection model
        self.face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.classifier = load_model('model.h5')
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

        # Create GUI elements
        self.label = ttk.Label(root, text="Emotion Detection", font=("Helvetica", 16))
        self.label.pack(pady=10)

        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        self.image_button = ttk.Button(root, text="Process Image", command=self.process_image)
        self.image_button.pack(pady=10)

        self.video_button = ttk.Button(root, text="Start Video", command=self.start_live_detection)
        self.video_button.pack(pady=10)

        self.quit_button = ttk.Button(root, text="Quit", command=self.root.destroy)
        self.quit_button.pack(pady=10)

    def process_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            image = cv2.imread(file_path)
            result_image = self.detect_emotion(image)
            self.display_frame(result_image)

    def start_live_detection(self):
        cap = cv2.VideoCapture(0)

        def process_frame():
            ret, frame = cap.read()
            if ret:
                frame_with_detection = self.detect_emotion(frame)
                self.display_frame(frame_with_detection)
                root.after(10, process_frame)  # Adjust delay if needed
            else:
                cap.release()
                cv2.destroyAllWindows()

        process_frame()

    def detect_emotion(self, frame):
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = self.classifier.predict(roi)[0]
                label = self.emotion_labels[prediction.argmax()]
                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

    def display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(frame)

        self.canvas.configure(width=frame.width(), height=frame.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=frame)
        self.canvas.image = frame  # Keep reference to avoid garbage collection

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectionApp(root)
    root.mainloop()
