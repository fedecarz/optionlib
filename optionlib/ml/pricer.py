import torch
import joblib
import numpy as np
from optionlib.ml.model import OptionPricer

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class MLPricer:
    
    @staticmethod
    def price(option):

        model = OptionPricer()
        model.load_state_dict(torch.load(os.path.join(BASE_DIR, "training", "model_weights.pt")))       # load trained weights

        model.eval()                # switches from training to "prediction mode"
        
        S = option.S
        K = option.K
        T = option.T
        r = option.r
        sigma = option.sigma
        q = option.q
        option_type = 1 if option.option_type == "call" else 0
        
        features = [S, K, T, r, sigma, q, option_type]

        features = np.array(features).reshape(1,-1)         # reshape to 2D array for scaler

        scaler = joblib.load(os.path.join(BASE_DIR, "training", "scaler.pkl"))      # load fitted scaler

        features = scaler.transform(features)               # apply same normalization as training

        X = torch.FloatTensor(features)

        with torch.no_grad():
            prediction = model(X)

        return prediction.item()


