import datetime
import streamlit as st
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import json
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns




def group_enseigne(name):
    name = str(name).lower()
    if  "access" in name:
        return "Total Access"  
    if "total" in name  :
        return "Total"  
    elif "carrefour" in name:
        return "Carrefour"
    elif "leclerc" in name:
        return "E.Leclerc"
    elif "inter" in name:
        return "Intermarché"
    elif "bp" in name:
        return "BP"
    elif "auchan" in name:
        return "Auchan"
    elif "géant" in name:
        return "Géant"
    elif "avia" in name:
        return "Avia"
    elif "esso" in name:
        return "ESSO"
    elif ("système u" in name) or "super" in name or "u express" in name :
        return "Super U"
    else:
        return name


# Title of the app
st.title("Analyse des stations-service : Carrefour vs concurrents")
 

 # Load data
prices = pd.read_csv('./DataTD/Prix_2024.csv')
stations = pd.read_csv('./DataTD/Infos_Stations.csv')



# Display data
st.subheader("CSV1: Prix")
st.dataframe(prices.head())
st.subheader("CSV2: Stations")
st.dataframe(stations.head())

# Step 1: Filter stations with type 'R'
st.header("Étape 1: Filtrer les stations routières")
stations = stations[stations['Type'] == 'R']
st.write(f"Nombre de stations routières : {len(stations)}")
st.dataframe(stations)


# Step 2: Filter enseignes with > 100 stations
st.header("Étape 2: Garder les enseignes importantes")
enseigne_counts = stations['Enseignes'].value_counts()
valid_enseignes = enseigne_counts[enseigne_counts > 100].index

stations = stations[stations['Enseignes'].isin(valid_enseignes)]
stations = stations[~stations['Enseignes'].isin(['Indépendant', 'Inconnu', 'Sans enseigne','Indépendant sans enseigne'])]
st.write(f"Enseignes conservées : {list(valid_enseignes)}")
st.dataframe(stations)


# Step 3: Handle outliers in prices
st.header("Étape 3: Gestion des valeurs aberrantes")
for col in ['Gazole', 'SP95', 'SP98', 'E10', 'E85', 'GPLc']:
    Q1 = prices[col].quantile(0.25)
    Q3 = prices[col].quantile(0.75)
    prices[col] = np.where((prices[col] < Q1) & (prices[col] != 0), Q1, prices[col])
    prices[col] = np.where((prices[col] > Q3 ) & (prices[col] != 0.0) , Q3, prices[col])
st.write("Valeurs aberrantes corrigées")
st.dataframe(prices.head())

# Step 4: Group by enseignes
st.header("Étape 4: Regrouper les enseignes")
stations['Enseignes'] = stations['Enseignes'].apply(group_enseigne)
st.write("Regroupement des enseignes terminé")
st.dataframe(stations)



df = pd.merge(prices, stations, on='ID', how='inner')

df["Groupe Enseigne"] = df["Enseignes"].apply(group_enseigne)