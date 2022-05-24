import pandas as pd
import numpy as np

# A script that edits .txt file like it was a .csv file

# Load the .txt file

filez = open('./OBCI_22_clean.txt', 'r+')

# Function that converts a .txt file to a .csv file

def convertz():
    # Read the .txt file
    data = filez.read()
    # Split the data into a list of lines
    lines = data.split('\n')
    # Create a list of lists
    new_lines = []
    # Loop through the lines
    for line in lines:
        # Split the line into a list of strings
        new_line = line.split('\t')
        # Add the new list to the list of lists
        new_lines.append(new_line)
    # Create a dataframe from the list of lists
    df = pd.DataFrame(new_lines)
    # Save the dataframe as a .csv file
    df.to_csv('./OBCI_22_clean.csv', index=False, header=False)

convertz()
