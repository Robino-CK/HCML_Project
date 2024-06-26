import pandas as pd

from sklearn.model_selection import train_test_split


def load_nasa():
    df = pd.read_csv('res/data/nasa.csv')
    # Drop columns with irrelevant information, repetitions (such as KM and Miles) and static values (such as 'Orbiting Body' and 'Equinox')
    df = df.drop(['Neo Reference ID', 'Name', 'Orbit ID', 'Close Approach Date',
                            'Epoch Date Close Approach', 'Orbit Determination Date', 'Est Dia in KM(max)', 'Est Dia in M(min)', 'Est Dia in M(max)', 'Est Dia in Miles(min)'
                ,'Est Dia in Miles(max)', 'Est Dia in Feet(min)', 'Est Dia in Feet(max)', 
                'Relative Velocity km per hr', 'Miles per hour', 'Miss Dist.(lunar)', 
                'Miss Dist.(kilometers)', 'Miss Dist.(miles)', 'Orbiting Body', 'Equinox'] , axis = 1)
    #Convert target values to binary
    df['Hazardous'] = df['Hazardous'].astype(int)
    X = df.drop(columns=['Hazardous']) #Features
    y = df['Hazardous'] #Target

    return X, y



def load_gas():
    df = pd.read_csv('res/data/Gas_Turbine/gt_2011.csv')
    
    

def prepare_data(X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
    # Get mean and standard deviation from train data
    X_means = x_train.mean()

    X_std = x_train.std()

    # Normalize train and test data (Centering, unit variance)
    x_train = x_train.apply(lambda x: (x-X_means)/X_std, axis = 1)
    x_test = x_test.apply(lambda x: (x-X_means)/X_std, axis = 1)
    return x_train, x_test, y_train, y_test

def normalize_data(values):
    ebm_values = []
    min_ebm = min(values)
    max_ebm = max(values)
    for v in values:
        ebm_values.append((v-min_ebm)/(max_ebm-min_ebm))
    return ebm_values


def get_data():
    X , y = load_nasa()
    return prepare_data(X, y)


def data_selected_features():
    X,y = load_nasa()
    X = X[["Minimum Orbit Intersection", "Absolute Magnitude", "Est Dia in KM(min)", "Eccentricity"]]
    return prepare_data(X, y)