from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# Main regression method, which can use any regression written below
def line_fit(df):
    # Alter Dataframe to to fit timeline, add PriceDateHour
    #df.drop(1142, inplace=True)
    df.drop(df.index[len(df)-24:], inplace=True)




    listo = np.array(df['EnergyPGE'].tolist())
    dateko = np.array(df['PriceDate']).tolist()

    for i in range(len(listo)):
        if listo[i] >= 120:
            #df.drop(i, inplace=True)
            df.loc[i, 'EnergyCISO'] = df.iloc[i-1]['EnergyCISO']


    pdh = range(len(df['PriceDate']))
    df['PDH'] = pdh


    factorios = ['LoadCISO', 'NetDemandCISO', 'WindCISO', 'SolarCISO']
    factorios = ['LoadCISO', 'NetDemandCISO']
    # Get Load, Demand, Solar, Wind such that they are inputs
    inputs = np.asarray(df[factorios])

    # Get MCE, MCC, MCL for target data
    outputsMCE = np.asarray(df['EnergyPGE'])
    outputsMCC = np.asarray(df['CongestCISO'])
    outputsMCL = np.asarray(df['LossCISO'])

    # Where the outputs will be stored
    TOTALPRICES = [0] * len(outputsMCC)
    # FORMAT: [[Importances], Absolute, Squared, Root]
    TOTALMCE = [[0]*len(factorios), 0, 0, 0]
    TOTALMCC = [[0]*len(factorios), 0, 0, 0]
    TOTALMCL = [[0]*len(factorios), 0, 0, 0]

    # Amount of Trials
    TRIALS = 1

    # Run through all trials
    for trial in range(TRIALS):
        print(trial)

        # Get MCEs, add to the totals
        pmce, importances, absolute, squared, root = random_forest(inputs, outputsMCE, "MCE Importance")
        add(TOTALMCE, [importances, absolute, squared, root])
        print("MCE")

        # Get MCCs, add to the totals
        pmcc, importances, absolute, squared, root = random_forest(inputs, outputsMCC, "MCC Importance")
        add(TOTALMCC, [importances, absolute, squared, root])
        print("MCC")

        # Get MCLs, add to the totals
        pmcl, importances, absolute, squared, root = random_forest(inputs, outputsMCL, "MCL Importance")
        add(TOTALMCL, [importances, absolute, squared, root])
        print("MCL")

        # Calculate prices for single trial, then add them to total prices
        pprices = [e + c + l for e, c, l in zip(pmce, pmcc, pmcl)]
        TOTALPRICES = [t + p for t, p in zip(TOTALPRICES, pprices)]

    # Print a buffer, calculate mean price
    """print()
    for i in range(len(TOTALPRICES)):
        TOTALPRICES[i] = TOTALPRICES[i] / float(TRIALS)"""

    # Calculate mean of each MCE, MCC, MCL
    TOTALMCE = divide(TOTALMCE, TRIALS)
    TOTALMCC = divide(TOTALMCC, TRIALS)
    TOTALMCL = divide(TOTALMCL, TRIALS)

    # Print all variables
    print(TOTALMCE)
    print(TOTALMCC)
    print(TOTALMCL)
    print(TOTALPRICES)

    # Print all three for prices
    print('Mean Absolute Error:', metrics.mean_absolute_error(df['EnergyCISO'], TOTALPRICES), end=', ')
    print('Mean Squared Error:', metrics.mean_squared_error(df['EnergyCISO'], TOTALPRICES), end=', ')
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(df['EnergyCISO'], TOTALPRICES)))

    # Set up plot
    fig, ax = plt.subplots()








    # Graph as many times as needed
    while True:

        #Graph MCE, MCC, MCL
        #if input("CHOOSE: ") == "y":
        graph_importance(TOTALMCE[0], "MCE Importance")
        #graph_importance(TOTALMCC[0], "MCC Importance")
        #graph_importance(TOTALMCL[0], "MCL Importance")

        # Plot calculated prices
        #if input("CHOOSE 2: ") == "y":
        ax.plot(df['PDH'], TOTALPRICES, label='Calculated')

        # Plot real prices
        #if input("CHOOSE 3: ") == "y":
        ax.plot(df['PDH'], df['EnergyCISO'], label='Real')

        # Actually plot
        try:
            ax.legend()
            plt.xlabel('Time')
            plt.ylabel('Prices')
            plt.title('Prices vs Time')
            plt.show()
        finally:
            print("DIFFERENT SETTINGS")

        # Quit if desired
        #if input("CHOOSE 4: ") == "q":
        break

    return 0

# The Random Forest regreessor
def random_forest(inputs, outputs, title):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.33, shuffle=True)

    # Create regressor, with 1000 estimators
    regressor = RandomForestRegressor(n_estimators=1000, random_state=42)

    # Fit data, predict with inputs to get outputs, get importances
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(inputs)
    importances = regressor.feature_importances_

    # Calculate metrics
    absolute = metrics.mean_absolute_error(outputs, y_pred)
    squared = metrics.mean_squared_error(outputs, y_pred)
    root = np.sqrt(squared)

    # Return
    return y_pred, importances, absolute, squared, root

# Linear Regressor
def linear_regression(inputs, outputs):
    #Split Data, fit, and teest
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.33, shuffle=True)
    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)
    print('Score: ', reg.score(X_test, y_test))
    print('Weights: ', reg.coef_)

# Bayesian Regressor
def bayesian_regression(inputs, outputs):
    # Split Data, fit, and teest
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.33, shuffle=True)
    reg = linear_model.BayesianRidge()
    reg.fit(X_train, y_train)
    print('Score: ', reg.score(X_test, y_test))
    print('Weights: ', reg.coef_)

# Ridge Regressor
def ridge(inputs, outputs):
    # Split Data, fit, and teest
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.33, shuffle=True)
    reg = linear_model.Ridge(alpha=.5)
    reg.fit(X_train, y_train)
    print('Score: ', reg.score(X_test, y_test))
    print('Weights: ', reg.coef_)

#Support Vector Regressor
def support_vector(inputs, outputs):
    # Split Data, fit, and teest
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.33, shuffle=True)
    reg = svm.SVR()
    reg.fit(X_train, y_train)
    print('Score: ', reg.score(X_test, y_test))



# Add a trials numbers to the total output
def add(array, numbers):
    for i in range(len(array[0])):
        array[0][i] += numbers[0][i]
    array[1] += numbers[1]
    array[2] += numbers[2]
    array[3] += numbers[3]

# Calculate means by dividing by them by the amount of trials
def divide(array, number):
    for i in range(len(array[0])):
        array[0][i] = array[0][i] / float(number)
    array[1] = array[1] / float(number)
    array[2] = array[2] / float(number)
    array[3] = array[3] / float(number)
    return array

# Graph the importances
def graph_importance(importances, title):
    # Create plot, plot importances with correct labels
    fig, ax = plt.subplots()
    ax.bar(['LOAD', 'DEMAND'], importances)
    ax.set_title(title)
    plt.show()


# Convert array to floats
def to_float(array):
    def f(x):
        return np.float(x)

    f2 = np.vectorize(f)
    return f2(array)


# Run main
if __name__ == "__main__":
    linear_regression("Main")
