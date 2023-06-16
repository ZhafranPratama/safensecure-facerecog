from ....core.logging import logger
from ..services.liverecog_service import LiveRecogService

# Module of an endpoint
class liveRecog:
    def __init__(self):
        pass

    def get_prediction(self, path):
        try:
            result = LiveRecogService.process(path)
            return result

        except Exception as e:
            logger.error('Error analysing an image :', e)
            return {"person": None, "image": None, "error-status": 1, "error-message": f"Error analysing an image: {e}"}