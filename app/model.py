from tensorflow.keras.models import load_model

class Model:
    def __init__(self):
        self.model = load_model(r'D:\Work_Place\Personal_Projects\Deepfake_Detection_Model\app\Deepfake_classif.h5')
        
    def prediction(self,input_pair):
        """Make model prediction from [rgb_input, dft_input]."""
        pred_prob = self.model.predict(input_pair)
        
        return pred_prob
    