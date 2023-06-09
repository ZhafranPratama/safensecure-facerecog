from ....core.logging import logger
from ..load_models import cv2, face_recognition, os, shutil, datetime, numpy as np, math, requests, tqdm, sys, Image, ImageOps, ImageEnhance, pickle, load_model, kerasImagePreprocess, kerasVGGFaceUtils
from ..load_models import models

CWD = os.getcwd()

# Module specific business logic (will be use for endpoints)
class RecogService:
    def __init__(self):
        pass

    def process(self, image):
        # Get time now for filename
        timeNow = self.getTimeNow()

        count = 1
        filename = f"{CWD}/data/output/recog/{timeNow}/{count}/data/input.jpeg"

        tmpcount = 1
        while os.path.exists(filename):
            filename = f"{CWD}/data/output/recog/{timeNow}/{tmpcount}/data/input.jpeg"
            count = tmpcount
            tmpcount += 1

        if not os.path.exists(f"{CWD}/data/output/recog/{timeNow}/"):
            os.mkdir(f"{CWD}/data/output/recog/{timeNow}/")
        if not os.path.exists(f"{CWD}/data/output/recog/{timeNow}/{count}/"):
            os.mkdir(f"{CWD}/data/output/recog/{timeNow}/{count}/")
        if not os.path.exists(f"{CWD}/data/output/recog/{timeNow}/{count}/data/"):
            os.mkdir(f"{CWD}/data/output/recog/{timeNow}/{count}/data/")

        # Save the image that is sent from the request and reject if filename is not valid
        with open(filename, "wb") as f:
            if image.filename.split(".")[-1].lower() not in ["jpg", "png", "jpeg", "heif"]:
                logger.warning("Filename not supported")
                return {"path_frame": None, "path_result": None, "result": None, "error_message": "Filename not supported", "status": 0}
            else:
                shutil.copyfileobj(image.file, f)
                logger.info(f"Saving image to {filename}")

        frame = cv2.imread(filename, cv2.IMREAD_COLOR)

        filenameDatas = {"timeNow": timeNow, "id": filename.split(f"{timeNow}/")[1].split("/data")[0]}

        filenames, confidences = models.getFaceCoordinates(frame, filenameDatas)

        if len(filenames) == 0:
            logger.info("API return success with exception: No face detected. Files removed")
            os.remove(filename)
            return {"path_frame": None, "path_result": None, "result": None, "error_message": "No face detected", "status": 0}
        
        resultRaw = []
        for currentFilename in filenames:
            print(currentFilename)
            try:
                resultRaw.append(self.recogFaceRecog(f"{CWD}/data/output/recog/{currentFilename}"))
            except:
                resultRaw.append(self.recogVGG(f"{CWD}/data/output/recog/{currentFilename}", count))

        frameNames = (i.split("/")[-1].split(".")[0] for i in filenames)
        
        result = {}
        for i, frameName in enumerate(frameNames):
            userDetected = resultRaw[i]
            if userDetected == "Unknown":
                result.update({frameName: "Unknown"})
            else:
                result.update({frameName: f"{userDetected}"})
        
        JSONFilename = f"{CWD}/data/output/recog/{timeNow}/{count}/data/face.json"

        with open(JSONFilename, "w") as f:
            f.write(str(result))

        logger.info("API return success. Request fulfilled.")
        return {"path_frame": filenames, "path_result": JSONFilename.split("output/")[1], "result": result, "status": 1}

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
                tmpFaceNames.append([IDdetected.split(".")[0], f"{confidence}%"])
        faceNames = tmpFaceNames

        return faceNames[0][0]
    
    def recogVGG(self, filename: str, requestFolderCount: int):
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
        
recogService = RecogService()