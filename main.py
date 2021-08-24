import pandas
import db_pull, regressions, data_formatter

def main():
    # Print all columns
    #pandas.set_option('display.max_columns', None)

    # Get data from database
    print("START")
    output = db_pull.get_data()

    # Alter the data slightly
    print("DATA")
    data_formatter.generate_demand(output)

    print("FORMAT")
    data_formatter.format(output)

    # Fitting the data using regressions
    print("DEMAND")
    regressions.line_fit(output)
    print("FITTED")

if __name__ == "__main__":
    main()
