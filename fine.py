import pandas as pd
import psycopg2
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Connect to the database
conn = psycopg2.connect(host='localhost', port='5432', dbname='fines', user='postgres', password='password')
cur = conn.cursor()

# Read the fines.csv file into a Pandas DataFrame
fines_df = pd.read_csv('fines.csv')

# Define a function to choose 3 fines
def choose_fines(keyword, fines_df):
    # Get the fines for the given keyword
    fines_df_filtered = fines_df[fines_df['Industry Keywords'].str.contains(keyword)]

    # If there are no fines for the given keyword, choose 3 random fines
    if len(fines_df_filtered) == 0:
        fines_df_filtered = fines_df.sample(3)

    # Choose 3 fines from the filtered DataFrame
    fines_chosen = fines_df_filtered.sample(3)

    # Return the chosen fines
    return fines_chosen

# Read the input.csv file into a Pandas DataFrame
input_df = pd.read_csv('input.csv')

# Create a new DataFrame to store the chosen fines
fines_picked_df = pd.DataFrame()

# Iterate over the input DataFrame and choose 3 fines for each keyword
for i in range(len(input_df)):
    keyword = input_df.iloc[i, 0]
    fines_chosen = choose_fines(keyword, fines_df)

    # Add the chosen fines to the new DataFrame
    fines_picked_df = fines_picked_df.append(fines_chosen)

# Export the chosen fines DataFrame to a CSV file
fines_picked_df.to_csv('fines_fine_pick.csv', index=False)

# Close the database connection
cur.close()
conn.close()