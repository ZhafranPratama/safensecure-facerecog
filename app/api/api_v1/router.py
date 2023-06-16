# API core module for all endpoints
from fastapi import APIRouter
from .endpoints.facerecog_endpoint import Recog
from .endpoints.liverecog_endpoint import liveRecog
from fastapi import UploadFile, File, Form

router = APIRouter(
    prefix='/api/v1',
    responses = {
        404: {'description': 'Not Found'}
    }
)

@router.post('/')
async def faceRecog(file: UploadFile = File(...)):
    recog = Recog()
    result = recog.get_prediction(file)

    return result

@router.post('/live')
async def faceRecog(path: str = Form(...)):
    recog = liveRecog()
    result = recog.get_prediction(path)

    return result