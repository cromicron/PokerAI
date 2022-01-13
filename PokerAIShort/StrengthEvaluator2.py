import pandas as pd
import numpy as np
#import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras.models import load_model
from keras import backend as K

#create lookup dictonary for preflop hands
modelsFolder = 'trainedModels/strengthEvaluator'
dfPreflop = pd.read_csv('strengthPreflop.csv', index_col = False)
lookupPreflop = {}
for i in dfPreflop.index:
    lookupPreflop[(dfPreflop.loc[i, 'val1'],dfPreflop.loc[i, 'val2'],dfPreflop.loc[i, 'suited'])] = (dfPreflop.loc[i, 'nwin'], dfPreflop.loc[i, 'nloose'])
    
flopModel = load_model(modelsFolder+'strengthFlop2.h5', custom_objects={"K": K})
turnModel = load_model(modelsFolder+'strengthTurn2.h5', custom_objects={"K": K})
riverModel = load_model(modelsFolder+'strengthRiver2.h5', custom_objects={"K": K})

class StrengthEvaluator:    
    #store the current hand strenghth    
    def evaluate(self, hand):
        #hand as list of tuples
        
        if len(hand) == 2:
            handSorted = hand.sort()
            #check if suited
            if hand[0][1] == hand[1][1]:
                suited = 1
            else:
                suited = 0
            
            self.pwin = lookupPreflop[hand[0][0], hand[1][0], suited][0]
            self.ploose = lookupPreflop[hand[0][0], hand[1][0], suited][1]
            self.pwinAvg = 0.4795
            self.plooseAvg = 0.4795
            self.stdWinAvg = 0.1074            
            return
        
        hand[0:2] = sorted(hand[0:2])
        hand[2:] = sorted(hand[2:])
        observation = np.array([hand]).flatten()
        if len(hand) == 5:
            predict = flopModel.predict(np.array([observation]))
            self.pwin, self.ploose, self.pwinAvg, self.plooseAvg, self.stdWinAvg = predict[0][0], predict[0][1], predict[0][2], predict[0][3], predict[0][4]
        elif len(hand) == 6:
            predict = turnModel.predict(np.array([observation]))
            self.pwin, self.ploose, self.pwinAvg, self.plooseAvg, self.stdWinAvg = predict[0][0], predict[0][1], predict[0][2], predict[0][3], predict[0][4]
        
        else:
            predict = riverModel.predict(np.array([observation]))
            self.pwin, self.ploose, self.pwinAvg, self.plooseAvg, self.stdWinAvg = predict[0][0], predict[0][1], predict[0][2], predict[0][3], predict[0][4]
