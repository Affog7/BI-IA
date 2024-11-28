import streamlit as st
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import json
import math
import plotly.express as px

# Chargement des fichiers CSV
stations = pd.read_csv('./DataTD/Infos_Stations.csv')
prices = pd.read_csv('./DataTD/Prix_2024.csv')

# Fonction pour regrouper les enseignes
def group_enseigne(name):
    name = str(name).lower()
    if "access" in name:
        return "Total Access"
    if "total" in name:
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
    elif "système u" in name or "super" in name or "u express" in name:
        return "Super U"
    else:
        return name

# Titre de l'application
st.title("Analyse des stations-service : Carrefour vs concurrents")

# Étape 1 : Conserver uniquement les stations routières
stations = stations[stations['Type'] == 'R']

# Étape 2 : Garder les enseignes importantes
enseigne_counts = stations['Enseignes'].value_counts()
valid_enseignes = enseigne_counts[enseigne_counts > 100].index
stations = stations[stations['Enseignes'].isin(valid_enseignes)]
stations = stations[~stations['Enseignes'].isin(['Indépendant', 'Inconnu', 'Sans enseigne', 'Indépendant sans enseigne'])]

# Étape 3 : Gestion des valeurs aberrantes
for col in ['Gazole', 'SP95', 'SP98', 'E10', 'E85', 'GPLc']:
    Q1 = prices[col].quantile(0.25)
    Q3 = prices[col].quantile(0.75)
    prices[col] = np.where((prices[col] < Q1) & (prices[col] != 0), Q1, prices[col])
    prices[col] = np.where(prices[col] > Q3, Q3, prices[col])

# Étape 4 : Regrouper les enseignes
stations['Enseignes'] = stations['Enseignes'].apply(group_enseigne)

# Étape 5 : Séparer les données en deux groupes
carrefour_stations = stations[stations['Enseignes'] == 'Carrefour']
other_stations = stations[stations['Enseignes'] != 'Carrefour']

# Étape 6 : Trouver les stations concurrentes dans un rayon de 10 km
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    R = 6371.0
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

competitors_dict = {}
carrefour_coords = carrefour_stations[['ID', 'Latitude', 'Longitude']]
other_coords = other_stations[['ID', 'Latitude', 'Longitude']]

for _, c_station in carrefour_coords.iterrows():
    c_id = c_station['ID']
    c_lat, c_lon = c_station['Latitude'] / 1e5, c_station['Longitude'] / 1e5
    competitors = []
    for _, o_station in other_coords.iterrows():
        o_id = o_station['ID']
        o_lat, o_lon = o_station['Latitude'] / 1e5, o_station['Longitude'] / 1e5
        distance = haversine(c_lat, c_lon, o_lat, o_lon)
        if distance <= 10:
            competitors.append(o_id)
    competitors_dict[c_id] = competitors

# Étape 7 : Comparaison des prix
# Étape 7 : Comparaison des prix
results = []
for date in prices['Date'].unique():
    daily_prices = prices[prices['Date'] == date]
    for product in ['Gazole', 'SP95', 'SP98', 'E10', 'E85', 'GPLc']:
        for _, c_station in carrefour_stations.iterrows():
            c_id = c_station['ID']
            
            # Vérification des données pour éviter l'erreur d'index
            c_price_row = daily_prices[daily_prices['ID'] == c_id][product]
            if c_price_row.empty:
                continue  # Passer si aucune donnée n'est disponible
            
            c_price = c_price_row.values[0]
            competitors = competitors_dict.get(c_id, [])
            comp_prices = daily_prices[daily_prices['ID'].isin(competitors)][product]

            results.append({
                'Date': date,
                'Produit': product,
                'Carrefour_Station_ID': c_id,
                'Inférieur': sum(comp_prices < c_price),
                'Égal': sum(comp_prices == c_price),
                'Supérieur': sum(comp_prices > c_price),
            })

results_df = pd.DataFrame(results)


# Étape 8 : Filtre interactif par date et visualisation
st.header("Étape 8 : Visualisation des résultats selon une date choisie")

# Sélecteur de date
unique_dates = sorted(prices['Date'].unique())
selected_date = st.selectbox("Choisissez une date", unique_dates)

# Filtrer les résultats pour la date choisie
filtered_results = results_df[results_df['Date'] == selected_date]

# Affichage des données filtrées
st.subheader(f"Résultats pour la date : {selected_date}")
st.dataframe(filtered_results)

# Barplot interactif
fig = px.bar(
    filtered_results,
    x="Produit",
    y=["Inférieur", "Égal", "Supérieur"],
    title=f"Comparaison des prix pour Carrefour et concurrents ({selected_date})",
    labels={"value": "Nombre de stations", "variable": "Comparaison"},
    barmode="group",
)
st.plotly_chart(fig)

# Téléchargement des résultats pour la date choisie
csv_file = f"comparison_{selected_date}.csv"
filtered_results.to_csv(csv_file, index=False)
st.download_button("Télécharger les résultats", data=open(csv_file).read(), file_name=csv_file)
