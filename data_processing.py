import pandas as pd

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