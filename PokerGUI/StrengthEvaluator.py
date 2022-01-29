import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

#create lookup dictonary for preflop hands
dfPreflop = pd.read_csv('strengthPreflop.csv', index_col = False)
lookupPreflop = {}
for i in dfPreflop.index:
    lookupPreflop[(dfPreflop.loc[i, 'val1'],dfPreflop.loc[i, 'val2'],dfPreflop.loc[i, 'suited'])] = (dfPreflop.loc[i, 'nwin'], dfPreflop.loc[i, 'nloose'])
flopModel = modelRiver = keras.models.load_model('strengthFlop.h5')
turnModel = modelRiver = keras.models.load_model('strengthTurn.h5')
riverModel = modelRiver = keras.models.load_model('strengthRiver.h5')

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
            
            pwin = lookupPreflop[hand[0][0], hand[1][0], suited][0]
            ploose = lookupPreflop[hand[0][0], hand[1][0], suited][1]
            self.pwin, self.ploose = pwin, ploose
            return
        
        hand[0:2] = sorted(hand[0:2])
        hand[2:] = sorted(hand[2:])
        observation = np.array([hand]).flatten()
        if len(hand) == 5:
            predict = flopModel.predict(np.array([observation]))
            self.pwin, self.ploose = predict[0][0], predict[0][1]
        elif len(hand) == 6:
            predict = turnModel.predict(np.array([observation]))
            self.pwin, self.ploose = predict[0][0], predict[0][1]
        
        else:
            predict = riverModel.predict(np.array([observation]))
            self.pwin, self.ploose= predict[0][0], predict[0][1]
