import base64
import cv2

def image_to_base64(img_rgb):
    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode("utf-8")