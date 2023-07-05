from ....core.logging import logger
from ..load_models import cv2, face_recognition, os, shutil, datetime, numpy as np, math, requests, tqdm, sys, Image, ImageOps, ImageEnhance, pickle, load_model, kerasImagePreprocess, kerasVGGFaceUtils, uuid
from ..load_models import models

CWD = os.getcwd()

# Module specific business logic (will be use for endpoints)
class liveRecogService:
    def __init__(self):
        pass

    def process(self, path):
        # Get time now for filename
        timeNow = self.getTimeNow()

        filename = f"{CWD}/data/{path}"

        count = 1
        while os.path.exists(f"{CWD}/data/output/live/{timeNow}/{count}/data/"):
            count += 1

        if not os.path.exists(f"{CWD}/data/output/live/{timeNow}/"):
            os.mkdir(f"{CWD}/data/output/live/{timeNow}/")
        if not os.path.exists(f"{CWD}/data/output/live/{timeNow}/{count}/"):
            os.mkdir(f"{CWD}/data/output/live/{timeNow}/{count}")
        if not os.path.exists(f"{CWD}/data/output/live/{timeNow}/{count}/data/"):
            os.mkdir(f"{CWD}/data/output/live/{timeNow}/{count}/data/")


        frame = cv2.imread(filename, cv2.IMREAD_COLOR)

        filenameDatas = {"timeNow": timeNow, "count": count}

        filenames, confidences = self.getFaceCoordinates(frame, filenameDatas)

        if len(filenames) == 0:
            logger.info("API return success with exception: No face detected. Files removed")
            os.remove(filename)
            return {"person": None, "image": None, "error_message": "No face detected", "status": 0}
        
        resultRaw = []
        for currentFilename in filenames:
            try:
                resultRaw.append(self.recogFaceRecog(f"{CWD}/data/output/{currentFilename}"))
            except:
                resultRaw.append(self.recogVGG(f"{CWD}/data/output/{currentFilename}"))

        result = []
        print(resultRaw)
        for index, i in enumerate(filenames):
            userDetected = resultRaw[index]
            if userDetected == "Unknown":
                result.append({"person": "Unknown", "image" : f"{filenames[index]}"})
            else:
                result.append({"person": f"{userDetected}", "image" : f"{filenames[index]}"})
        
        JSONFilename = f"{CWD}/data/output/live/{timeNow}/{count}/data/face.json"

        with open(JSONFilename, "w") as f:
            f.write(str(result))

        logger.info("API return success. Request fulfilled.")
        return {"output" : result, "status": 1}

    def getTimeNow(self):
        # before: %d-%b-%y.%H-%M-%S
        return datetime.datetime.now().strftime("%Y%m%d")

    def recogFaceRecog(self, filename: str):
        logger.info("Recognizing faces into user IDs")
        # Read image as cv2
        frame = cv2.imread(filename, cv2.IMREAD_COLOR)

        frame = models.resize(filename, 480)

        faceNames = list(self.getFaceNames(frame))

        tmpFaceNames = []
        for i in faceNames:
            IDdetected = i.split("-")[0]
            if IDdetected == "Unknown (0%)":
                IDdetected = "Unknown"
                confidence = 0
            else:
                confidence = i.split("jpeg (")[1].split("%")[0]
            # Threshold confidence of 85% for the API to return
            if float(confidence) > 85 or IDdetected != "Unknown":
                tmpFaceNames.append([IDdetected, f"{confidence}%"])
        faceNames = tmpFaceNames

        return faceNames[0][0]
    
    def recogVGG(self, filename: str):
        logger.info("Recognizing faces into user IDs")

        # Set the dimensions of the image
        imageWidth = 224 
        imageHeight = 224

        # load the training labels
        faceLabelFilename = f'{CWD}/ml-models/training-models/face-labels.pickle'
        with open(faceLabelFilename, "rb") as \
            f: class_dictionary = pickle.load(f)

        class_list = [value for _, value in class_dictionary.items()]

        # Load the image
        imgtest = cv2.imread(filename, cv2.IMREAD_COLOR)
        
        # Load model
        today = datetime.datetime.now().strftime("%Y%m%d")
        trainedFilename = f'{CWD}/ml-models/training-models/{today}-trained.h5'
        if not os.path.exists(trainedFilename):
            logger.warning("PROGRAM IS ENCODING WHEN SOMEONE IS SENDING REQUEST.")
            self.encodeFaces()
        
        model = load_model(trainedFilename)

        facesDetected = []

        # Resize the detected face to 224 x 224
        size = (imageWidth, imageHeight)
        resized_image = cv2.resize(imgtest, size)
 
        # Preparing the image for prediction
        x = kerasImagePreprocess.img_to_array(resized_image)
        x = np.expand_dims(x, axis=0)
        x = kerasVGGFaceUtils.preprocess_input(x, version=1)

        # Predicting
        predicted_prob = model.predict(x)
        facesDetected.append(class_list[predicted_prob[0].argmax()])

        return facesDetected[0]

    def getFaceNames(self, frame):
        # Find all the faces and face encodings in the image
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(models.known_face_encodings, face_encoding)
            name = "Unknown"
            confidence = '0%'

            # Calculate the shortest distance to face
            face_distances = face_recognition.face_distance(models.known_face_encodings, face_encoding)

            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = models.known_face_names[best_match_index]
                confidence = self.faceConfidence(face_distances[best_match_index])
            
            face_names.append(f'{name} ({confidence})')

        return face_names
    
    def faceConfidence(self, face_distance, face_match_threshold=0.6):
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)

        if face_distance > face_match_threshold:
            return str(round(linear_val * 100, 2)) + '%'
        else:
            value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
            return str(round(value, 2)) + '%'
        
    def getFaceCoordinates(self, frame, filenameDatas):
        logger.info("Grabbing faces detected from input image")

        timeNow = filenameDatas["timeNow"]
        count = filenameDatas["count"]
        detector = cv2.FaceDetectorYN.create(f"{CWD}/ml-models/face_detection_yunet/face_detection_yunet_2022mar.onnx", "", (224, 224))

        height, width, channels = frame.shape

        # Set input size
        detector.setInputSize((width, height))
        # Getting detections
        channel, faces = detector.detect(frame)
        faces = faces if faces is not None else []

        boxes = []
        confidences = []
        filenames = []
        numFace = 1
        
        for face in faces:
            box = list(map(int, face[:4]))
            boxes.append(box)
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            faceCropped = frame[y:y + h, x:x + w]
            # boxes = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            ### SEMENTARA MASIH TANPA FILTERING MINIMUM PIXEL SHAPE 50
            # if w >= 50 and h >= 50 and x >= 0 and y >= 0:
            try:
                filename = f"{CWD}/data/output/live/{timeNow}/{count}/data/{uuid.uuid4()}-{numFace}.jpeg"
                filenames.append(filename.split("/output/")[1])
                cv2.imwrite(filename, faceCropped)
                cv2.imwrite(filename, models.resize(filename, 360))
                numFace += 1
                    
                confidence = face[-1]
                confidence = "{:.2f}%".format(confidence*100)

                confidences.append(confidence)
            except:
                pass

        logger.info(f"Face grab success. Got total faces of {len(filenames)}")
        return (filenames, confidences)
        
LiveRecogService = liveRecogService()