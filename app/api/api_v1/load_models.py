# for load machine learning models
import datetime
import cv2
import face_recognition
import os
import shutil
import numpy
import math
import requests
from tqdm import tqdm
import sys
from PIL import Image, ImageOps, ImageEnhance
import json

# Fixing vggface import error
# filename = "/usr/local/lib/python3.8/dist-packages/keras_vggface/models.py"
# text = open(filename).read()
# open(filename, "w+").write(text.replace('keras.engine.topology', 'keras.utils.layer_utils'))

from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras_vggface.vggface import VGGFace
import pickle
from keras.preprocessing import image as kerasImagePreprocess
from keras_vggface import utils as kerasVGGFaceUtils
from keras.models import load_model

from ...core.logging import logger

# testing pr

CWD = os.getcwd()

class Models:
    def __init__(self):
        self.face_encodings = []
        self.known_face_encodings = []
        self.known_face_names = []

    def encodeFacesFaceRecog(self):
        # Update dataset before encoding
        self.updateDatasetFaceRecog()

        # Encoding faces (Re-training for face detection algorithm)
        logger.info("FaceRecog : Encoding Faces... (This may take a while)")
        for image in tqdm(os.listdir(f'{CWD}/data/dataset/FaceRecog'), file=sys.stdout):
            face_image = face_recognition.load_image_file(f'{CWD}/data/dataset/FaceRecog/{image}')
            try:
                face_encoding = face_recognition.face_encodings(face_image)[0]
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(image)
            except IndexError:
                pass

        
        logger.info("FaceRecog : Encoding Done!")

    def encodeFacesVGG(self):
        today = datetime.datetime.now()
        if os.path.exists(f"{CWD}/ml-models/training-models/{today.strftime('%Y%m%d')}-trained.h5"):
            logger.info("ENCODING AND UPDATE SKIPPED. MODEL EXISTS.")
            return None
        else:
            try:
                today = today - datetime.timedelta(days=1)
                os.remove(f"{CWD}/ml-models/training-models/{today.strftime('%Y%m%d')}-trained.h5")
            except FileNotFoundError:
                logger.info("VGG : First time training, creating new initial train file.")

        # Update dataset before encoding
        self.updateDatasetVGG()

        # Encoding faces (Re-training for face detection algorithm)
        logger.info("VGG : Encoding Faces... (This may take a while)")
        
        # NOTE: UNCOMMENT THIS LINE IF YOU WANT TO USE GPU INSTEAD OF CPU
        # tf.config.list_physical_devices('gpu')

        DATASET_DIRECTORY = f"{CWD}/data/dataset/VGG"

        # Preprocess dataset
        trainDatagen = ImageDataGenerator(
            preprocessing_function=preprocess_input)

        # Setup for dataset training
        trainGenerator = \
            trainDatagen.flow_from_directory(
            DATASET_DIRECTORY,
            target_size=(224,224),
            color_mode='rgb',
            batch_size=32,
            class_mode='categorical',
            shuffle=True)

        # Get list of classes
        trainGenerator.class_indices.values()
        NO_CLASSES = len(trainGenerator.class_indices.values())

        # Initiate training model
        baseModel = VGGFace(include_top=False,
        model='vgg16',
        input_shape=(224, 224, 3))
        # NOTE: IF ERROR, UNCOMMENT. IF NOT ERROR, DELETE.
        # baseModel.summary()

        # Setup first layers
        x = baseModel.output
        x = GlobalAveragePooling2D()(x)

        x = Dense(1024, activation='relu')(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)

        # Setup final layer with sigmoid activation
        preds = Dense(NO_CLASSES, activation='softmax')(x)

        # Create a new model with the base model's original input and the new model's output
        model = Model(inputs = baseModel.input, outputs = preds)
        model.summary()

        # Don't train the first 19 layers - 0..18
        for layer in model.layers[:19]:
            layer.trainable = False

        # Train the rest of the layers - 19 onwards
        for layer in model.layers[19:]:
            layer.trainable = True

        # Compling the model
        model.compile(optimizer='Adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        # MAIN TRAINING
        model.fit(trainGenerator,
        batch_size = 32,
        verbose = 1,
        epochs = 50)

        # Create HDF5 file
        today = datetime.datetime.now().strftime("%Y%m%d")
        model.save(f'{CWD}/ml-models/training-models/{today}-trained.h5')

        classDictionary = trainGenerator.class_indices
        classDictionary = {
            value:key for key, value in classDictionary.items()
        }

        # Save the class dictionary to pickle
        faceLabelFilename = f'{CWD}/ml-models/training-models/face-labels.pickle'
        with open(faceLabelFilename, 'wb') as f:
            pickle.dump(classDictionary, f)
        
        logger.info("VGG : Encoding Done!")

    def updateDatasetFaceRecog(self):
        logger.info("FaceRecog : Updating datasets... (This may took a while)")

        # APITOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InJlemFhckBrYXplZS5pZCIsImlhdCI6MTY4MzI3NjIzNH0.mYMzLDf5oZtREuKMya5JgN8rwr7v5Hs_wS13eaM4Hrw"
        # r = requests.get("http://103.150.120.58:3003/api/profile/list-photo", headers={'Authorization': 'Bearer ' + APITOKEN})

        # datas = r.json()["data"]

        with open(f'{CWD}/data/response_data_fr.json', 'r') as j:
            datas = json.loads(j.read())

        for data in tqdm(datas, file=sys.stdout):
            userID = data["_id"]
            url = data["photo"]

            r = requests.get(url)

            filename = f'{CWD}/data/dataset/FaceRecog/{userID}.jpeg'
            
            # Save grabbed image to {CWD}/data/faces/
            with open(filename, 'wb') as f:
                f.write(r.content)
            
            self.imgAugmentation(filename)

        logger.info("FaceRecog : Datasets updated!")

    def updateDatasetVGG(self):
        logger.info("VGG : Updating datasets... (This may took a while)")

        # APITOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InJlemFhckBrYXplZS5pZCIsImlhdCI6MTY3NTgyMTY2Mn0.eprZiRQUjiWjbfZYlbziT6sXG-34f2CnQCSy3yhAh6I"
        # r = requests.get("http://103.150.87.245:3001/api/profile/list-photo", headers={'Authorization': 'Bearer ' + APITOKEN})

        # datas = r.json()["data"]

        with open(f'{CWD}/data/response_data_fr.json', 'r') as j:
            datas = json.loads(j.read())

        for data in tqdm(datas, file=sys.stdout):
            userID = data["_id"]
            url = data["photo"]

            r = requests.get(url)

            foldername = f'{CWD}/data/dataset/VGG/{userID}'

            if not os.path.exists(foldername):
                os.mkdir(foldername)

            filename = f"{foldername}/{userID}.jpeg"
            
            # Save grabbed image to {CWD}/data/faces/
            with open(filename, 'wb') as f:
                f.write(r.content)
            
            self.imgAugmentation(filename)

        logger.info("VGG : Datasets updated!")

    def imgAugmentation(self, img):
        try:
            frame = Image.open(img)
            frame = frame.convert("RGB")
            frame = numpy.array(frame)
            detector = cv2.FaceDetectorYN.create(f"{CWD}/ml-models/face_detection_yunet/face_detection_yunet_2022mar.onnx", "", (224, 224))
            height, width, channels = frame.shape
            detector.setInputSize((width, height))
            channel, faces = detector.detect(frame)
            faces = faces if faces is not None else []
            boxes = []
            for face in faces:
                box = list(map(int, face[:4]))
                boxes.append(box)
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                faceCropped = frame[y:y + h, x:x + w]
            if len(boxes) > 1:
                print("More than 1 face detected. Only choosing the first face that got detected")
            if len(boxes) != 0:
                cv2.imwrite(img, cv2.cvtColor(faceCropped, cv2.COLOR_BGR2RGB))
        except Exception as e:
            logger.error(f"ERROR - {str(e)}. Filename: {img}")

        # Read image
        input_img = Image.open(img)
        input_img = input_img.convert('RGB')
        # Flip Image
        img_flip = ImageOps.flip(input_img)
        img_flip.save(f"{img.split('.jpeg')[0]}-flipped.jpeg")
        # Mirror Image 
        img_mirror = ImageOps.mirror(input_img)
        img_mirror.save(f"{img.split('.jpeg')[0]}-mirrored.jpeg")
        # Rotate Image
        img_rot1 = input_img.rotate(30)
        img_rot1.save(f"{img.split('.jpeg')[0]}-rotated1.jpeg")
        img_rot2 = input_img.rotate(330)
        img_rot2.save(f"{img.split('.jpeg')[0]}-rotated2.jpeg")
        # Adjust Brightness
        enhancer = ImageEnhance.Brightness(input_img)
        im_darker = enhancer.enhance(0.5)
        im_darker.save(f"{img.split('.jpeg')[0]}-darker1.jpeg")
        im_darker2 = enhancer.enhance(0.7)
        im_darker2.save(f"{img.split('.jpeg')[0]}-darker2.jpeg")
        enhancer = ImageEnhance.Brightness(input_img)
        im_darker = enhancer.enhance(1.2)
        im_darker.save(f"{img.split('.jpeg')[0]}-brighter1.jpeg")
        im_darker2 = enhancer.enhance(1.5)
        im_darker2.save(f"{img.split('.jpeg')[0]}-brighter2.jpeg")

    def getFaceCoordinates(self, frame, filenameDatas):
        logger.info("Grabbing faces detected from input image")

        timeNow = filenameDatas["timeNow"]
        id = filenameDatas["id"]
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
        count = 1
        
        for face in faces:
            box = list(map(int, face[:4]))
            boxes.append(box)
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            faceCropped = frame[y:y + h, x:x + w]

            ### SEMENTARA MASIH TANPA FILTERING MINIMUM PIXEL SHAPE 50
            # if w >= 50 and h >= 50 and x >= 0 and y >= 0:
            filename = f"{CWD}/data/output/{timeNow}/{id}/frame/frame{str(count).zfill(3)}.jpeg"
            if not os.path.exists(f"{CWD}/data/output/{timeNow}/{id}/frame/"):
                os.mkdir(f"{CWD}/data/output/{timeNow}/{id}/frame/")
            filenames.append(filename.split("output/")[1])
            cv2.imwrite(filename, faceCropped)
            cv2.imwrite(filename, self.resize(filename, 360))
            count += 1
                
            confidence = face[-1]
            confidence = "{:.2f}%".format(confidence*100)

            confidences.append(confidence)

        logger.info(f"Face grab success. Got total faces of {len(filenames)}")
        return (filenames, confidences)
    
    def resize(self, filename: str, resolution: int):
        frame = cv2.imread(filename)
        if frame.shape[0] != resolution or frame.shape[1] != resolution:
            return cv2.resize(frame, (0, 0), fx=1-(frame.shape[1]-resolution)/frame.shape[1], fy=1-(frame.shape[1]-resolution)/frame.shape[1])
        else:
            return frame