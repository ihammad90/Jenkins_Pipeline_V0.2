import pandas as pd
import numpy as np

from preprocessors import Pipeline
import config

from pipeline import pipeline



def train_function():
    a = np.genfromtxt('train_score.csv', delimiter=',')
    return a

def test_function():
    a = np.genfromtxt('test_score.csv', delimiter=',')
    return a


if __name__ == '__main__':
    
    # load data set
    #data = pd.read_csv(config.PATH_TO_DATASET)
    
    #pipeline.ads(data)
    pipeline.train()
    print('model performance')
    pipeline.evaluate_model()
    print()
    pipeline.r2_value_train()
    pipeline.r2_value_test()
    
    #print('Some predictions:')
    #preditions = pipeline.predict(data)
    #print(preditions)