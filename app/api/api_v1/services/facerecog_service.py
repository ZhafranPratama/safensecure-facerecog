from ....core.logging import logger
from ..load_models import cv2, face_recognition, os, shutil, datetime, numpy as np, math, requests, tqdm, sys, Image, ImageOps, ImageEnhance, pickle, load_model, kerasImagePreprocess, kerasVGGFaceUtils
from ..load_models import Models

CWD = os.getcwd()

models = Models()
models.encodeFacesFaceRecog()
models.encodeFacesVGG()

# Module specific business logic (will be use for endpoints)
class RecogService:
    def __init__(self):
        pass

    def process(self, image):
        # Get time now for filename
        timeNow = self.getTimeNow()

        count = 1
        filename = f"{CWD}/data/output/{timeNow}/{count}/data/input.jpeg"

        tmpcount = 1
        while os.path.exists(filename):
            filename = f"{CWD}/data/output/{timeNow}/{tmpcount}/data/input.jpeg"
            count = tmpcount
            tmpcount += 1

        if not os.path.exists(f"{CWD}/data/output/{timeNow}/"):
            os.mkdir(f"{CWD}/data/output/{timeNow}/")
        if not os.path.exists(f"{CWD}/data/output/{timeNow}/{count}/"):
            os.mkdir(f"{CWD}/data/output/{timeNow}/{count}/")
        if not os.path.exists(f"{CWD}/data/output/{timeNow}/{count}/data/"):
            os.mkdir(f"{CWD}/data/output/{timeNow}/{count}/data/")

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
            try:
                resultRaw.append(self.recogFaceRecog(f"{CWD}/data/output/{currentFilename}"))
            except:
                resultRaw.append(["Unknown", "0%"])

        frameNames = (i.split("/")[-1].split(".")[0] for i in filenames)
        
        result = {}
        for i, frameName in enumerate(frameNames):
            userDetected = resultRaw[i][0]
            confidence = resultRaw[i][1]
            if userDetected == "Unknown":
                result.update(self.processVGG(filename, count))
            else:
                result.update({frameName: f"{userDetected} :: {confidence}"})
        
        JSONFilename = f"{CWD}/data/output/{timeNow}/{count}/data/face.json"

        with open(JSONFilename, "w") as f:
            f.write(str(result))

        logger.info("API return success. Request fulfilled.")
        return {"path_frame": filenames, "path_result": JSONFilename.split("output/")[1], "result": result, "status": 1}
    
    def processVGG(self, image, count):
        # Get time now for filename
        timeNow = self.getTimeNow()
        facesDetected, frameNames = self.recogVGG(image, count)

        if len(facesDetected) == 0:
            logger.info("API return success with exception: No face detected. Files removed")
            os.remove(image)
            return {"path_frame": None, "path_result": None, "result": None, "error_message": "No face detected", "status": 0}
        
        result = {}
        for i, frameName in enumerate(frameNames):
            result.update({frameName.split("/frame/")[1].split(".")[0]: f"{facesDetected[i]}"})

        JSONFilename = f"{CWD}/data/output/{timeNow}/{count}/data/face.json"

        with open(JSONFilename, "w") as f:
            f.write(str(result))

        logger.info("API return success. Request fulfilled.")
        return result

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

        return [faceNames[0][0], faceNames[0][1]]
    
    def recogVGG(self, filename: str, requestFolderCount: int):
        logger.info("Recognizing faces into user IDs")

        # Set the dimensions of the image
        imageWidth, imageHeight = (224, 224)

        # load the training labels
        faceLabelFilename = f'{CWD}/ml-models/training-models/face-labels.pickle'
        with open(faceLabelFilename, "rb") as \
            f: class_dictionary = pickle.load(f)

        class_list = [value for _, value in class_dictionary.items()]

        # Detecting faces
        detector = cv2.FaceDetectorYN.create(f"{CWD}/ml-models/face_detection_yunet/face_detection_yunet_2022mar.onnx", "", (224, 224))

        # Load the image
        imgtest = cv2.imread(filename, cv2.IMREAD_COLOR)
        image_array = np.array(imgtest, "uint8")

        # Get the faces detected in the image
        height, width, channels = imgtest.shape
        detector.setInputSize((width, height))
        channel, faces = detector.detect(imgtest)
        faces = faces if faces is not None else []
        boxes = []

        # Load model
        today = datetime.datetime.now().strftime("%Y%m%d")
        trainedFilename = f'{CWD}/ml-models/training-models/{today}-trained.h5'
        if not os.path.exists(trainedFilename):
            logger.warning("PROGRAM IS ENCODING WHEN SOMEONE IS SENDING REQUEST.")
            self.encodeFaces()
        
        model = load_model(trainedFilename)

        facesDetected = []
        frames = []
        confidence = []

        count = 1
        for face in faces:
            box = list(map(int, face[:4]))
            boxes.append(box)
            face_x = box[0]
            face_y = box[1]
            face_w = box[2]
            face_h = box[3]
            # Resize the detected face to 224 x 224
            size = (imageWidth, imageHeight)
            roi = image_array[face_y: face_y + face_w, face_x: face_x + face_h]
            resized_image = cv2.resize(roi, size)

            frame = f"{CWD}/data/output/{today}/{requestFolderCount}/frame"
            if not os.path.exists(frame):
                os.mkdir(frame)
                
            frame += f"/frame{str(count).zfill(3)}.jpeg"
            count += 1
            
            cv2.imwrite(frame, resized_image)

            frames.append(frame.split("output/")[1])

            # Preparing the image for prediction
            x = kerasImagePreprocess.img_to_array(resized_image)
            x = np.expand_dims(x, axis=0)
            x = kerasVGGFaceUtils.preprocess_input(x, version=1)

            # Predicting
            predicted_prob = model.predict(x)
            facesDetected.append(class_list[predicted_prob[0].argmax()])

        return facesDetected, frames

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