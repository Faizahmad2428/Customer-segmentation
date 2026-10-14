import pandas as pd

# Load your Excel file
df = pd.read_csv('online-retail-dataset.csv')

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], dayfirst=False, errors='coerce')

# Reformat to 'YYYY-MM-DD HH:MM:SS' — this adds dashes instead of slashes
df['InvoiceDate'] = df['InvoiceDate'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Preview
print(df['InvoiceDate'].head())