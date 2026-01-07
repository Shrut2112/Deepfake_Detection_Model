from tensorflow.keras.models import load_model
from pathlib import Path

class Model:
    def __init__(self):
        Base_DIR = Path(__file__).resolve().parent
        MODEL_PATH = Base_DIR / "Deepfake_classif.h5"
        self.model = load_model(MODEL_PATH)
        
    def prediction(self,input_pair):
        """Make model prediction from [rgb_input, dft_input]."""
        pred_prob = self.model.predict(input_pair)
        
        return pred_prob
    