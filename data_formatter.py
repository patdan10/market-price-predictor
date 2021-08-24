import numpy as np

# Function to generate new dataframe points
def generate_demand(df):
    df['NetDemandPGE'] = df['LoadPGE'] - df['WindPGE'] - df['SolarPGE']
    df['NetDemandSCE'] = df['LoadSCE'] - df['WindSCE'] - df['SolarSCE']
    df['WindCISO'] = df['WindSCE'] + df['WindPGE']
    df['SolarCISO'] = df['SolarSCE'] + df['SolarPGE']
    df['NetDemandCISO'] = df['LoadCISO'] - df['WindPGE'] - df['SolarPGE'] - df['WindSCE'] - df['SolarSCE']
    df['EnergyPGE'] = df['PricePGE'] - df['CongestPGE'] - df['LossPGE']
    df['EnergySCE'] = df['PriceSCE'] - df['CongestSCE'] - df['LossSCE']
    df['EnergyCISO'] = df[['EnergyPGE', 'EnergySCE']].mean(axis=1)
    df['CongestCISO'] = df[['LossPGE', 'LossPGE']].mean(axis=1)
    df['LossCISO'] = df[['LossPGE', 'LossPGE']].mean(axis=1)

def format(df):
    df['EnergyCISO'] = df['EnergyCISO'].map(lambda x: float(np.floor(x) // 2 * 2))
    df['LoadCISO'] = df['LoadCISO'].map(lambda x: float(np.floor(x) // 1000 * 1000))
    return df
