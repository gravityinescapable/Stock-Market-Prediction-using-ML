import pandas as pd

file_path = 'NIFTY_50_23_years.csv'
df = pd.read_csv(file_path)

# Reverse the order of the data 
reversed_df = df.iloc[::-1]

# Write the reversed DataFrame back to the CSV file
reversed_df.to_csv('reversed_' + file_path, index=False)
