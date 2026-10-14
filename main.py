# main.py

import pandas as pd
import numpy as np
import mysql.connector
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def connect_db():
    return mysql.connector.connect(
        host='your_host',
        user='your_user',
        password='your_password',
        database='your_database'
    )

def fetch_rfm_data():
    conn = connect_db()
    query = "SELECT * FROM rfm_view;"  # You need to have a view or table with R, F, M values
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def preprocess_rfm(df):
    rfm = df[['Recency', 'Frequency', 'Monetary']]
    scaler = StandardScaler()
    scaled_rfm = scaler.fit_transform(rfm)
    return scaled_rfm, df

def apply_kmeans(scaled_data, original_df, n_clusters=4):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    original_df['Cluster'] = model.fit_predict(scaled_data)
    return model, original_df
