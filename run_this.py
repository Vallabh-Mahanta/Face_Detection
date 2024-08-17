from detector import Detector

if __name__ == "__main__":
    modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
    videoPath = 0  # Set to 0 for default webcam
    classFile = "coco.names"

    detector = Detector(modelURL, classFile)
    detector.predictVideo(videoPath)
