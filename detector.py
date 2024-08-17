import cv2
import os
import tensorflow as tf
import numpy as np
from gtts import gTTS  # Importing gTTS
import threading

from tensorflow.python.keras.utils.data_utils import get_file

# Seed for reproducibility
np.random.seed(20) 

class Detector:
    def __init__(self, modelURL, classFilePath):
        self.engine = None
        self.model = None
        self.cacheDir = None
        self.modelName = None
        self.classesList = None
        self.colorList = None
        self.threshold = 0.5
    
        self.readClasses(classFilePath) 
        self.downloadModel(modelURL)
        self.loadModel()
    def readClasses(self, classFilePath):
        with open(classFilePath, 'r') as f:
            self.classesList = f.read().splitlines()
            # Generate random colors for each class
            self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))
            print(len(self.classesList), len(self.colorList))
            
    def downloadModel(self, modelURL):
        fileName = os.path.basename(modelURL)
        self.modelName = fileName[:fileName.index('.')]
        self.cacheDir = "./pretrained_models"
        os.makedirs(self.cacheDir, exist_ok=True)
        
        # Check if the model is already downloaded
        if not os.path.exists(os.path.join(self.cacheDir, "checkpoints", self.modelName, "saved_model")):
            get_file(fname=fileName,
                     origin=modelURL,cache_dir=self.cacheDir, cache_subdir="checkpoints", extract=True)

    def loadModel(self):
        if self.model is None:
            print("Loading Model... "+self.modelName)
            tf.keras.backend.clear_session()
            self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints", self.modelName, "saved_model"))
            print("Model "+ self.modelName + " _Loaded SucccesFully...👍🏻")
    
    def releaseModel(self):
        if self.model is not None:
            tf.keras.backend.clear_session()
            self.model = None
    
    def detectObjects(self, image):
        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor = inputTensor[tf.newaxis,...]

        detections = self.model(inputTensor)
               
        bboxs = detections['detection_boxes'][0].numpy()
        classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
        classScores = detections['detection_scores'][0].numpy()
        
        imH, imW, imC = image.shape
        bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=50,
                    iou_threshold=self.threshold, score_threshold=self.threshold)
        
        print(bboxIdx)
        
        detected_objects = []
        if len(bboxIdx) != 0: 
            for i in bboxIdx:
                bbox = tuple(bboxs[i].tolist())
                classConfidence = (100*classScores[i])
                classIndex = classIndexes[i]
                
                classLabelText = self.classesList[classIndex].upper() #textUpp
                classColor = self.colorList[classIndex]

                displayText = '{}: {}%'.format(classLabelText, classConfidence)
                ymin, xmin, ymax, xmax = bbox

                xmin, xmax, ymin, ymax = (xmin * imW, xmax * imW, ymin*imH, ymax*imH)
                xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=2)
                cv2.putText(image, displayText, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)
                
                detected_objects.append({
                    'class': classLabelText,
                    'confidence': classConfidence,
                    'bbox': (xmin, ymin, xmax, ymax)
                })
                                        
                
        return image, detected_objects

    def speak_object(self, detected_objects, center_line):
        for obj in detected_objects:
            class_name = obj['class']
            xmin, _, _, _ = obj['bbox']
            if xmin < center_line:
                # Detected object is on the left side, play speech in left earbud
                left_speech = gTTS(text=class_name, lang='en', slow=False)
                left_speech.save("left_output.mp3")
                os.system("afplay left_output.mp3")  # Adjust based on your OS and audio setup
            elif xmin > center_line:
                # Detected object is on the right side, play speech in right earbud
                right_speech = gTTS(text=class_name, lang='en', slow=False)
                right_speech.save("right_output.mp3")
                os.system("afplay right_output.mp3")  # Adjust based on your OS and audio setup
            else:
                # Detected object is in the center, play speech in both earbuds
                tts = gTTS(text=class_name, lang='en', slow=False)
                tts.save("output.mp3")
                os.system("afplay output.mp3")  # Adjust based on your OS and audio setup

    def predictVideo(self, videoPath):
        cap = cv2.VideoCapture(videoPath)
        frame_count = 0
        detected_objects = []  # Initialize detected_objects list

        if not cap.isOpened():
            print("Error: Unable to open video file.")
            return

        # Get the width and height of the video frame
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate the center of the frame
        center_line = width // 2

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Draw a red line at the center of the frame
            cv2.line(frame, (center_line, 0), (center_line, height), (0, 0, 255), thickness=2)

            # Perform object detection on each frame
            processed_frame, detected_objects = self.detectObjects(frame)

            # Display the updated detection results
            cv2.imshow("Result", processed_frame)

            # Speak the detected objects every 4 seconds
            if frame_count % 90 == 0:  # Assuming 30 frames per second
                threading.Thread(target=self.speak_object, args=(detected_objects, center_line), daemon=True).start()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Add code to initialize and use the Detector class
