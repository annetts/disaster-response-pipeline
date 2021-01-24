import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    ''' Function to load data in from csv files.
    INPUT - massages and categories file path
    OUTPUT - pandas dataframe of the data
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = pd.merge(messages, categories, left_on='id', right_on='id', how='inner')

    return df


def clean_data(df):
    ''' Function to clean the loaded data from undefined 
    and unnecessary rows.
    INPUT - dataframe
    OUTPUT - cleaned dataframe
    '''
    
    # create a dataframe of the 36 individual category columns
    categories = pd.DataFrame(df.categories.str.split(';', 36, expand=True))

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])

    categories.columns = category_colnames

    for column in categories.columns:
        # set each value to be the last character of the string
        categories[column] = categories[column].map(lambda row: row.split('-')[1])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # duplicateRowsDF = df[df.duplicated()]
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    ''' Function to save cleaned data to sqllite database.
    INPUT - dataframe, database file name 
    '''
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
