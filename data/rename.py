import pandas as pd

def rename_columns(csv_file_path, new_column_names):
    df = pd.read_csv(csv_file_path)
    if 'Index Name' in df.columns:
        del df['Index Name']
    df.rename(columns=new_column_names, inplace=True)
    df.to_csv(csv_file_path, index=False)

# Define the new column names
new_column_names = {
    "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    
}

csv_file_path = 'reversed_NIFTY_50_23_years.csv'
rename_columns(csv_file_path, new_column_names)
print("Columns renamed successfully.")
