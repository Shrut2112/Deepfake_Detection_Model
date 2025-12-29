from fastapi import FastAPI,UploadFile,File,HTTPException
from .model import Model
import numpy as np
from app.preprocess import get_dft_rgb_feat
import cv2
from .schema import PredictionResponse
from .utilis import image_to_base64
import logging
from app.core.logging import setup_logger

app = FastAPI()
model = Model()

setup_logger()
logger = logging.getLogger("deepfake_detection")

@app.post("/predict",response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    logger.info("Prediction request received")
    
    if file.content_type not in ["image/jpeg", "image/png"]:
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(400, "Only JPEG/PNG supported")
    
    try:
        file_r = await file.read()
        file_bytes = np.asarray(bytearray(file_r), dtype=np.uint8)
    
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        if img is None:
            logger.error("Failed to decode image")
            raise HTTPException(400, "Invalid image file")
    
        logger.info(f"Received file: {file.filename} with content type: {file.content_type}")
    
        rgb_feat, dft_feat, img_up,mag,phase,ela_chan = get_dft_rgb_feat(img)

        pred_prob= model.prediction([rgb_feat, dft_feat])
        classes = ['fake','real']    
        pred = float(pred_prob)
    
        y_val_pred = (pred >= 0.509)
        pred_label = classes[y_val_pred]
        certainty = pred if y_val_pred else (1 - pred)
        certainty_percent = certainty * 100
        logger.info(f"Prediction successful | probability={pred:.4f}")
        
    except Exception as e:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(500, "Internal server error")
    
    return PredictionResponse(
        prediction=pred_label,
        probability=pred,
        certainty=certainty_percent,
        og_image=image_to_base64(img_up[0]),
        ela_image=image_to_base64((ela_chan * 255).astype(np.uint8)),
        dft_image=image_to_base64(((mag * 255).astype(np.uint8)))
    )