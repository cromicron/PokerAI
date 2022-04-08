from StrengthEvaluator2 import flopModel, turnModel, riverModel, lookupPreflop
import numpy as np
import pandas as pd

def transformStateVillain(state_memory, handsVillain):
    #function takes in numpy arrays of hero-states, and villain hands (same order) and transofrms then into corresponding villain states
    vil = handsVillain
    state_memory_villain = state_memory.copy()
    state_memory_villain[:,0] = np.where(state_memory[:,0]==1, 0, 1)
    #state_memory_villain[:,1], state_memory_villain[:,2] = state_memory[:,2], state_memory[:,1]
    state_memory_villain[:,4], state_memory_villain[:,5] = vil[:,0,0], vil[:,1,0]
    state_memory_villain[:,6] = np.where(vil[:,0,1]!= vil[:,1,1], 0, 1)       
    state_memory_villain[:,10] == np.where(state_memory_villain[:,3]==0, -1, vil[:,0,1])
    state_memory_villain[:,11] == np.where(state_memory_villain[:,3]==0, -2, vil[:,1,1])

    #extract Hand-Information from state_memory_villain
    arrHands = np.array([state_memory_villain[:,4],state_memory_villain[:,10], state_memory_villain[:,5], 
                         state_memory_villain[:,11],state_memory_villain[:,-15], state_memory_villain[:,-14], 
                         state_memory_villain[:,-13], state_memory_villain[:,-12],
                         state_memory_villain[:,-11], state_memory_villain[:,-10],state_memory_villain[:,-9], 
                         state_memory_villain[:,-8],state_memory_villain[:,-7], state_memory_villain[:,-6], state_memory_villain[:,6]])
    
    #create a pandas data frame to simplify operations
    hands = pd.DataFrame(columns = ['val1', 'suit1', 'val2', 'suit2', 'f1val', 'f1suit', 'f2val', 'f2suit', 'f3val', 'f3suit', 'tval', 'tsuit', 'rval', 'rsuit', 'suited'])
    
    #import hands into dataframe
    hands[['val1', 'suit1', 'val2', 'suit2', 'f1val', 'f1suit', 'f2val', 'f2suit', 'f3val', 'f3suit', 'tval', 'tsuit', 'rval', 'rsuit','suited']] = arrHands.transpose().astype(int)
    
    #extract per street
    preflop = hands[['val1', 'val2','suited']][hands['suit1']<0]
    flop = hands[['val1', 'suit1', 'val2', 'suit2', 'f1val', 'f1suit', 'f2val', 'f2suit', 'f3val', 'f3suit']][(hands['suit1']>-1) & (hands['tval'] < 0)]
    turn = hands[['val1', 'suit1', 'val2', 'suit2', 'f1val', 'f1suit', 'f2val', 'f2suit', 'f3val', 'f3suit','tval','tsuit']][(hands['tval'] > -1) & (hands['rval'] < 0)]
    river = hands[['val1', 'suit1', 'val2', 'suit2', 'f1val', 'f1suit', 'f2val', 'f2suit', 'f3val', 'f3suit','tval','tsuit','rval','rsuit']][hands['rval'] > -1]
    lenPreflop, lenFlop, lenTurn, lenRiver = 0, 0, 0, 0
    #To lookup in lookuptable, we must generate tuple of values and suited
    if len(preflop) > 0:
        lenPreflop =1
        preflop['tuple']=preflop.apply(lambda row : (row['val1'], row['val2'], row['suited']), axis = 1)
    
    #and lookup in lookuptable -villain values are trivial
        preflop['pwin']=preflop.apply(lambda row : lookupPreflop[row['tuple']][0], axis = 1)
        preflop['ploose']=preflop.apply(lambda row : lookupPreflop[row['tuple']][1], axis = 1)
        preflop['pwinVil']= 0.4795
        preflop['plooseVil']= 0.4795
        preflop['stdWinVil']= 0.1074
        preflopResults = preflop[['pwin', 'ploose', 'pwinVil', 'plooseVil', 'stdWinVil']]
    #for postflop streets we run thorugh the neural network
    
    if len(flop) > 0:
        lenFlop = 1
        flop[['pwin', 'ploose', 'pwinVil', 'plooseVil', 'stdWinVil']] = flopModel.predict(np.array(flop))
        flopResults = flop[['pwin', 'ploose', 'pwinVil', 'plooseVil', 'stdWinVil']]
        
    if len(turn) > 0:
        lenTurn = 1
        turn[['pwin', 'ploose', 'pwinVil', 'plooseVil', 'stdWinVil']] = turnModel.predict(np.array(turn))
        turnResults = turn[['pwin', 'ploose', 'pwinVil', 'plooseVil', 'stdWinVil']]
        
    if len(river) > 0:
        lenRiver = 1
        river[['pwin', 'ploose', 'pwinVil', 'plooseVil', 'stdWinVil']] = riverModel.predict(np.array(river)) 
        riverResults = river[['pwin', 'ploose', 'pwinVil', 'plooseVil', 'stdWinVil']]
    
    #let's create a data Frame with the results
    streetFrames = []
    if lenPreflop == 1:
        streetFrames.append(preflopResults)
    if lenFlop == 1:
        streetFrames.append(flopResults)
    if lenTurn == 1:
        streetFrames.append(turnResults)
    if lenRiver ==1:
        streetFrames.append(riverResults)
        
    if len(streetFrames) > 1:
        results = pd.concat(streetFrames).sort_index()
    else:
        results = streetFrames[0].sort_index()
    #and add into state_memory_villain
    state_memory_villain[:,-5:] = results
    
    return state_memory_villain