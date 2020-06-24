from utils.feature_engineering import feature_engineering
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path, test_size=0.25, inference=False):
    input_data = pd.read_csv(path)
    engineered = feature_engineering(input_data)
    telcom = engineered[0]
    Id_col = engineered[2]
    target_col = engineered[3]

    cols    = [i for i in telcom.columns if i not in Id_col + target_col]
    
        ##seperating dependent and independent variables
    if inference == False:
        train, test = train_test_split(telcom, test_size = test_size ,random_state = 111)
        train_X = train[cols]
        train_Y = train[target_col]
        test_X  = test[cols]
        test_Y  = test[target_col]
        
        data_list = [train_X, train_Y, test_X, test_Y]
    else:
        data_list = [telcom[cols]]
    return data_list