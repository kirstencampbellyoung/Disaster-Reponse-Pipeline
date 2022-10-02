import sys

import pandas
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ Load in the messages and categories .csv files and convert them to pandas dataframes.
    :param messages_filepath: filepath string to messages file
    :param categories_filepath: filepath string to the categories file
    :return: messages: pandas dataframe of messages data
    :return: categories: pandas dataframe of categories data
    """

    messages = pandas.read_csv(messages_filepath)
    categories = pandas.read_csv(categories_filepath)

    return messages, categories


def clean_data(messages, categories):
    """ Merges the 2 datasets and cleans them up
    :param messages: pandas dataframe of messages data
    :param categories: pandas dataframe of categories data
    :return: clean pandas dataframe of mergerd messages and categories
    """

    #merge messages with categories
    df = pandas.merge(messages, categories, on='id')

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', n=-1, expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x.split('-')[0])

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])

        # convert column from string to numeric
        categories[column] = categories[column].astype(np.int64)

    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pandas.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(subset=None, keep='first', inplace=True)

    return df


def save_data(df, database_filename):
    """ Save the new dataframe into a SQL database
    :param df: processed dataframe
    :param database_filename: filename of the database
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('df', engine, index=False)


def main():
    """ Clean the data and load it into a database"""
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages, categories)

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