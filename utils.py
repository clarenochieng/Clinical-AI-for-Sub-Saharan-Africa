import pandas as pd

def find_unique_values(df):
    unique_values = set()
    for column in df.columns:
        unique_values.update(df[column].unique().tolist())
    return list(unique_values)

