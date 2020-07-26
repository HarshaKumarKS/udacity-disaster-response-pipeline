"""
ETL(Etract, Transform and Load) pipleline
Disaster Response Project

Paths
    1) CSV file of messages > disaster_messages.csv
    2) CSV file of categories > disaster_categories.csv
    3) Destination file of SQLite DB > disaster_response.db
"""

# Importing the necessary libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
 
def load_messages_with_categories(messages_filepath, categories_filepath):
    """
    Function to Load Messages Data with Categories 
    
    Inputs:
    
     Path to the CSV file of messages > messages_filepath
     Path to the CSV file of categories > categories_filepath
        
    Output:
        
     Data frame with combined messages and categories data > df
        
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #Merge messages and categories based on key value of id
    df = pd.merge(messages,categories,on='id')
    return df 

def clean_categories_data(df):
    """
    Function to Clean Categories Data 
    
    Arguments:
     
     Combined data of messages and categories > df
    
    Outputs:
    
     Data frame after cleaning up the categories column > df
     
    """
    
    # Spliting the categories column by making each value as a new column
    categories = df['categories'].str.split(pat=';',expand=True)
    
    # Select first row of the categories dataframe
    first_row = categories.iloc[[1]]
    
    # Use the selected first row of the dataframe as the new column name for categories
    new_col_name = [category_name.split('-')[0] for category_name in first_row.values[0]]
    
    # Rename the columns
    categories.columns = new_col_name
    
    
    for column in categories:
        
        # keep only the last charecter of the string value
        categories[column] = categories[column].str[-1]
        
        # String values are converted into integer values (0 or 1)
        categories[column] = categories[column].astype(np.int)
    
    #Drop the redundant categories column
    df = df.drop('categories',axis=1)
    
    # Add the new columns with the original data frame
    df = pd.concat([df,categories],axis=1)
    
    # drop the duplicate rows
    df = df.drop_duplicates()
    
    return df

def save_data_to_db(df, database_filename):
    """
    Function to Save Data to SQLite DB
    
    Arguments:
       
    SQLite database destination - >  database_filename
    
    """
    cre_eng = create_engine('sqlite:///'+ database_filename)
    table_name = database_filename.replace(".db","") + "_table"
    df.to_sql(table_name, cre_eng, index=False, if_exists='replace')

def main():
    """
    Main function primarily performs loding the data with catagories, cleaning and saving it into SQLite Database
  
    """
    
    #Check if the argument counts is equal to 4 and execute the ETL pipeline
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:] 

        print('Loading messages data from {} ...\nLoading categories data from {} ...'
              .format(messages_filepath, categories_filepath))
        
        df = load_messages_with_categories(messages_filepath, categories_filepath)

        print('Cleaning categories data ...')
        df = clean_categories_data(df)
        
        print('Saving data to SQLite Database : {}'.format(database_filepath))
        save_data_to_db(df, database_filepath)
        
        print('Cleaned data has been successfully saved to the database!')
    
    else: 
        print("Please give the proper arguments: \nSample Script Execution:\n\
> python process_data.py disaster_messages.csv disaster_categories.csv disaster_response_db.db \n\
Arguments Description: \n\
1) Path to the CSV file containing messages (e.g. disaster_messages.csv)\n\
2) Path to the CSV file containing categories (e.g. disaster_categories.csv)\n\
3) Path to SQLite destination database (e.g. disaster_response_db.db)")

if __name__ == '__main__':
    main()