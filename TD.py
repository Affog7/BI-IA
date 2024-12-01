import datetime
import streamlit as st
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import json
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# Read the CSV files
# csv1 = pd.read_csv('./DataTD/Infos_Stations.csv')
# csv2 = pd.read_csv('./DataTD/Prix_2024.csv')
# merged_df = pd.merge(df1, df2, on='ID')

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
# st.title("Analyse des stations-service : Carrefour vs concurrents")
 

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
# st.write(f"Enseignes conservées : {list(valid_enseignes)}")
# st.dataframe(stations)

# Step 3: Handle outliers in prices
st.header("Étape 3: Gestion des valeurs aberrantes")
for col in ['Gazole', 'SP95', 'SP98', 'E10', 'E85', 'GPLc']:
    Q1 = prices[col].quantile(0.25)
    Q3 = prices[col].quantile(0.75)
    prices[col] = np.where((prices[col] < Q1) & (prices[col] != 0), Q1, prices[col])
    prices[col] = np.where((prices[col] > Q3 ) & (prices[col] != 0.0) , Q3, prices[col])
st.write("Valeurs aberrantes corrigées")
st.dataframe(prices.head())

# # Step 4: Group by enseignes
# st.header("Étape 4: Regrouper les enseignes")
stations['Enseignes'] = stations['Enseignes'].apply(group_enseigne)
st.write("Regroupement des enseignes terminé")
st.dataframe(stations)



df = pd.merge(prices, stations, on='ID', how='inner')

df["Groupe Enseigne"] = df["Enseignes"].apply(group_enseigne)




stations_per_groupe = df.groupby("Groupe Enseigne")["ID"].nunique().reset_index()
stations_per_groupe.columns = ["Groupe Enseigne", "Nombre de Stations"]

# Graphique : Nombre de stations par groupe d'enseigne
fig = px.bar(
    stations_per_groupe,
    x="Groupe Enseigne",
    y="Nombre de Stations",
    title="Nombre de stations par groupe d'enseigne",
    labels={"Nombre de Stations": "Nombre de stations", "Groupe Enseigne": "Groupe d'enseigne"},
    color="Groupe Enseigne",
)

# Afficher les résultats dans Streamlit
st.title("Analyse des Performances par Groupe d'Enseignes")

st.subheader("1. Tableau des performances par groupe")
st.dataframe(stations_per_groupe)

st.subheader("2. Graphique : Nombre de stations par groupe d'enseigne")
st.plotly_chart(fig)

st.header("Étape 4: Regrouper les enseignes")
stations['Enseignes'] = stations['Enseignes'].replace({'Carrefour Market': 'Carrefour'})
st.dataframe(stations)

# # Step 5: Separate Carrefour data
# st.header("Étape 5: Séparer les données Carrefour et concurrents")
carrefour_stations = stations[stations['Enseignes'] == 'Carrefour']
other_stations = stations[stations['Enseignes'] != 'Carrefour']

# st.write("Stations Carrefour")
# st.dataframe(carrefour_stations)
# st.write("Autres stations")
# st.dataframe(other_stations)

import math

# Haversine function to calculate distance between two points
def haversine(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    # Earth's radius in km
    R = 6371.0
    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance



# Step 6: Find competitors within 10 km
 
from functools import lru_cache
# Étape 6 : Trouver les stations concurrentes dans un rayon de 10 km
st.header("Étape 6 : Trouver les stations concurrentes dans un rayon de 10 km")

@st.cache_data
def calculate_competitors(carrefour_coords, other_coords):
    competitors_dict = {}

    # Itérer sur les stations Carrefour
    for _, c_station in carrefour_coords.iterrows():
        c_id = c_station['ID']
        c_lat, c_lon = c_station['Latitude'] / 1e5, c_station['Longitude'] / 1e5
        competitors = []

        # Calculer les distances avec toutes les autres stations
        for _, o_station in other_coords.iterrows():
            o_id = o_station['ID']
            o_lat, o_lon = o_station['Latitude'] / 1e5, o_station['Longitude'] / 1e5
            distance = haversine(c_lat, c_lon, o_lat, o_lon)
            if distance <= 10:  # Dans un rayon de 10 km
                competitors.append(o_id)

        # Stocker les concurrents pour cette station
        competitors_dict[c_id] = competitors

    return competitors_dict

# Charger les données et calculer les concurrents
carrefour_coords = carrefour_stations[['ID', 'Latitude', 'Longitude']].copy()
other_coords = other_stations[['ID', 'Latitude', 'Longitude']].copy()
competitors_dict = calculate_competitors(carrefour_coords, other_coords)

# Afficher et télécharger le JSON
# st.write("Dictionnaire des concurrents (JSON généré)")
# # st.json(competitors_dict)
# st.download_button(
#     "Télécharger le JSON",
#     data=json.dumps(competitors_dict),
#     file_name="competitors.json"
# )

# Étape 7 : Comparaison des prix pour les concurrents

# ID, Enseigne, Produits




# st.header("Étape 7 : Comparaison des prix")

@st.cache_data
def compare_prices(selected_date, carrefour_stations, prices, competitors_dict):
    daily_prices = prices[prices['Date'] == selected_date]
    list_product_ = ['Gazole', 'SP95', 'SP98', 'E10', 'E85', 'GPLc']
    results = []

    for product in list_product_:
        for _, c_station in carrefour_stations.iterrows():
            c_id = c_station['ID']

            # Vérification si le produit est présent pour cette station
            c_price_data = daily_prices[daily_prices['ID'] == c_id]
            if c_price_data.empty or product not in c_price_data:
                continue  # Sauter cette station si pas de données

            # Obtenir le prix de Carrefour pour ce produit
            c_price = c_price_data[product].values[0]

            # Trouver les concurrents
            competitors = competitors_dict.get(c_id, [])
            comp_prices = daily_prices[daily_prices['ID'].isin(competitors)][product]

            # Récupérer le nom de l'enseigne Carrefour
            carrefour_name = (
                carrefour_stations[carrefour_stations["ID"] == c_id]["Enseignes"].iloc[0]
                if not carrefour_stations[carrefour_stations["ID"] == c_id].empty
                else "Inconnu"
            )

            # Ajouter les résultats pour cette station et ce produit
            results.append({
                'Date': selected_date,
                'Produit': product,
                'Carrefour_Station_ID': c_id,
                'Carrefour_Station_Name': carrefour_name,
                'Inférieur': sum(comp_prices < c_price),
                'Égal': sum(comp_prices == c_price),
                'Supérieur': sum(comp_prices > c_price),
            })

    return pd.DataFrame(results)

# Spécifier la date et comparer les prix
selected_date = "2024-03-15"
results_df = compare_prices(selected_date, carrefour_stations, prices, competitors_dict)

# Affichage des résultats
st.dataframe(results_df)

# Spécifier la date et comparer les prix
selected_date = "2024-03-15"
results_df = compare_prices(selected_date, carrefour_stations, prices, competitors_dict)

# Affichage des résultats
# st.dataframe(results_df)

# Téléchargement des résultats au format CSV
# csv_file = f"price_comparison_{selected_date}.csv"
# results_df.to_csv(csv_file, index=False)
# st.download_button("Télécharger les résultats (CSV)", data=open(csv_file).read(), file_name=csv_file)




# Step 8
# Visualisation des résultats (Étape 8) 

# Groupement des données par station et produit
grouped_df = results_df.groupby(['Carrefour_Station_ID', 'Produit'])[['Inférieur', 'Égal', 'Supérieur']].sum().reset_index()

# Transformation en format long pour les graphiques
df_pivoted = grouped_df.melt(
    id_vars=["Carrefour_Station_ID", "Produit"],
    value_vars=["Inférieur", "Égal", "Supérieur"],
    var_name="Mesures",
    value_name="Nombre"
)

# Associer les enseignes aux ID dans le DataFrame
df_pivoted = df_pivoted.merge(
    stations[['ID', 'Enseignes']],  # Associer via la table des stations
    left_on="Carrefour_Station_ID",
    right_on="ID",
    how="left"
)
df_pivoted.drop(columns=["ID"], inplace=True)  # Supprimer la colonne redondante

# Affichage des données pivotées
# st.dataframe(df_pivoted)

# # Tracé des barres empilées pour un produit spécifique
# st.header("Filtrer par Produit et Date")

# Sélection des dates disponibles
available_dates = prices["Date"].unique()
selected_date = st.selectbox("Sélectionnez une date", sorted(available_dates))

# Sélection des produits disponibles
available_products = df_pivoted["Produit"].unique()
selected_product = st.selectbox("Sélectionnez un produit", available_products)

# Filtrage des données en fonction de la sélection
filtered_results = results_df[(results_df["Date"] == selected_date) & (results_df["Produit"] == selected_product)]

# st.subheader(f"Résultats pour {selected_product} le {selected_date}")
# st.dataframe(filtered_results)

# # Ajout des enseignes des concurrents dans le DataFrame
# st.subheader("Barres Empilées : Comparaison des Prix par Enseigne des Concurrents")

# Création du mapping pour les enseignes des concurrents
competitor_enseigne_map = other_stations[['ID', 'Enseignes']].set_index('ID').to_dict()['Enseignes']

# Debug : Afficher le dictionnaire des enseignes concurrentes
# st.write("Mapping des enseignes concurrentes :")
# st.dataframe(pd.DataFrame.from_dict(competitor_enseigne_map, orient="index", columns=["Enseigne"]))

# Ajouter les enseignes des concurrents à chaque station Carrefour
results_df['Enseigne_Concurrents'] = results_df['Carrefour_Station_ID'].map(
    lambda x: [competitor_enseigne_map[comp_id] for comp_id in competitors_dict.get(x, []) if comp_id in competitor_enseigne_map]
)

# Déplier la colonne "Enseigne_Concurrents" pour un format tabulaire
exploded_results = results_df.explode('Enseigne_Concurrents')

# Agrégation des données par enseigne concurrente
grouped_by_enseigne = exploded_results.groupby(['Enseigne_Concurrents', 'Produit'])[['Inférieur', 'Égal', 'Supérieur']].sum().reset_index()

# Transformation des données pour Seaborn
df_pivoted_enseigne = grouped_by_enseigne.melt(
    id_vars=["Enseigne_Concurrents", "Produit"],
    value_vars=["Inférieur", "Égal", "Supérieur"],
    var_name="Mesures",
    value_name="Nombre"
)

# Filtrer les données pour le produit sélectionné
df_filtered_enseigne = df_pivoted_enseigne[df_pivoted_enseigne["Produit"] == selected_product]

# Création du graphique avec Seaborn
# fig, ax = plt.subplots(figsize=(12, 6))
# sns.barplot(
#     data=df_filtered_enseigne,
#     x='Enseigne_Concurrents', 
#     y='Nombre', 
#     hue='Mesures', 
#     dodge=False,  # Empilement
#     palette="Set2",
#     ax=ax
# )

# Mise en forme du graphique
# ax.set_title(f"Comparaison des prix par enseignes concurrentes pour '{selected_product}'", fontsize=16)
# ax.set_xlabel("Enseignes Concurrentes", fontsize=14)
# ax.set_ylabel("Nombre de Comparaisons", fontsize=14)
# ax.tick_params(axis='x', rotation=45)
# ax.legend(title="Mesures", fontsize=12)
# plt.tight_layout()

# Affichage du graphique dans Streamlit
# st.pyplot(fig)


## KPI

# Étape 1 : Ajouter une fonction pour calculer le prix moyen par enseigne
def calculate_avg_price(df, selected_date, product_column, enseignes_list):
    """Calculer le prix moyen pour chaque enseigne pour un produit donné."""
    # Filtrer les données pour la date sélectionnée
    daily_prices = df[df['Date'] == selected_date]
    
    # Créer un dictionnaire pour stocker les prix moyens par enseigne
    avg_prices = {}

    for enseigne in enseignes_list:
        # Filtrer les stations de l'enseigne
        enseigne_stations = daily_prices[daily_prices['Enseignes'] == enseigne]
        
        if not enseigne_stations.empty:
            # Calculer le prix moyen pour ce carburant
            avg_price = enseigne_stations[product_column].mean()
            avg_prices[enseigne] = round(avg_price, 2)  # Arrondir à 2 décimales

    return avg_prices

# Étape 2 : Créer un widget pour sélectionner la date
st.title("Indicateurs Clés de Performance par Enseigne")

# Liste des enseignes pré-calculées à partir de l'étape précédente
enseignes_list = ['Carrefour', 'Auchan', 'E.Leclerc', 'Total Access', 'Intermarché', 'Super U']

# Sélectionner une date parmi celles disponibles dans les données
selected_date_ = st.selectbox("Sélectionnez une date", sorted(available_dates), key="select_date_")

# Liste des types de carburant
list_product_ = ['Gazole', 'SP95', 'SP98', 'E10', 'E85', 'GPLc']

# Créer des colonnes pour afficher 3 tableaux par ligne
columns = st.columns(3)

# Compteur pour contrôler les colonnes
col_index = 0

# Pour chaque produit, calculer les prix moyens et afficher un tableau séparé
for product in list_product_:
    # Calculer les prix moyens pour chaque enseigne
    avg_prices = calculate_avg_price(df, selected_date_, product, enseignes_list)
    
    # Créer un DataFrame pour chaque produit
    product_df = pd.DataFrame(list(avg_prices.items()), columns=['Enseigne', 'Prix Moyen (€)'])
    
    # Afficher le tableau pour ce produit dans la colonne correspondante
    with columns[col_index]:
        st.subheader(f"Prix moyen pour {product} le {selected_date_}")
        st.dataframe(product_df.style.format({'Prix Moyen (€)': '€ {:.2f}'}))
    
    # Passer à la colonne suivante
    col_index += 1
    
    # Si nous avons atteint la troisième colonne, réinitialiser l'index
    if col_index == 3:
        col_index = 0  # Réinitialiser l'index pour recommencer la ligne suivante






# etape 9
import folium
from streamlit_folium import st_folium
from geopy.distance import geodesic
import matplotlib.dates as mdates

# Titre de la page
st.title("Carte Interactive des Stations Carrefour et Concurrents")

# Rayon de recherche en kilomètres (modifiable si nécessaire)
radius_km = 10

# Fonction pour afficher la carte interactive
def afficher_carte(selected_carrefour_id):
    # Données de la station Carrefour sélectionnée
    selected_station = carrefour_stations[carrefour_stations['ID'] == selected_carrefour_id].iloc[0]
    selected_coords = (selected_station['Latitude'] / 100000, selected_station['Longitude'] / 100000)

    # Calcul des stations concurrentes dans un rayon de 10 km
    concurrent_stations = other_stations.copy()
    concurrent_stations['Distance_km'] = concurrent_stations.apply(
        lambda row: geodesic(selected_coords, (row['Latitude'] / 100000, row['Longitude'] / 100000)).kilometers,
        axis=1
    )
    nearby_competitors = concurrent_stations[concurrent_stations['Distance_km'] <= radius_km]

    # Création de la carte Folium centrée sur la station Carrefour sélectionnée
    m = folium.Map(location=selected_coords, zoom_start=12)

    # Ajouter la station Carrefour sur la carte
    folium.Marker(
        location=selected_coords,
        popup=f"Carrefour : {selected_station['Enseignes']} ({selected_station['Ville']})",
        tooltip="Station Carrefour",
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)

    # Ajouter les stations concurrentes sur la carte
    for _, row in nearby_competitors.iterrows():
        folium.Marker(
            location=(row['Latitude'] / 100000, row['Longitude'] / 100000),
            popup=f"{row['Enseignes']} ({row['Ville']})",
            tooltip=row['Enseignes'],
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)

    # Afficher la carte et les détails
    st.write(f"**Station Carrefour sélectionnée : {selected_station['Enseignes']} ({selected_station['Ville']})**")
    st.write(f"**Nombre de concurrents dans un rayon de {radius_km} km : {len(nearby_competitors)}**")
    st_folium(m, width=800, height=600)

# Fonction pour afficher un tableau comparant les prix
def afficher_tableau_comparaison(df, selected_carrefour_id, product_column):
    # Données de la station Carrefour sélectionnée
    selected_station = carrefour_stations[carrefour_stations['ID'] == selected_carrefour_id].iloc[0]
    selected_coords = (selected_station['Latitude'] / 100000, selected_station['Longitude'] / 100000)
    
    # Filtrer les stations concurrentes dans un rayon de 10 km
    concurrent_stations = other_stations.copy()
    concurrent_stations['Distance_km'] = concurrent_stations.apply(
        lambda row: geodesic(selected_coords, (row['Latitude'] / 100000, row['Longitude'] / 100000)).kilometers,
        axis=1
    )
    nearby_competitors = concurrent_stations[concurrent_stations['Distance_km'] <= radius_km]
    
    # Ajouter les prix pour la station Carrefour
    st.dataframe(selected_station)
    carrefour_price = selected_station[product_column]
    
    # Créer un DataFrame avec les stations concurrentes et leurs prix
    competitor_prices = nearby_competitors[['Enseignes', 'Ville', product_column]]
    competitor_prices = competitor_prices.append(
        {'Enseignes': selected_station['Enseignes'], 'Ville': selected_station['Ville'], product_column: carrefour_price},
        ignore_index=True
    )
    
    # Trier les prix par ordre décroissant
    competitor_prices = competitor_prices.sort_values(by=product_column, ascending=False)
    
    # Afficher le tableau avec la ligne Carrefour en vert
    competitor_prices['Color'] = competitor_prices['Enseignes'].apply(
        lambda x: 'background-color: green; color: white' if x == selected_station['Enseignes'] else ''
    )
    
    st.subheader(f"Comparaison des prix pour {product_column} à {selected_station['Enseignes']} ({selected_station['Ville']})")
    st.dataframe(competitor_prices.style.applymap(lambda x: 'background-color: green; color: white' if x == selected_station['Enseignes'] else '', subset=['Enseignes']))

# Fonction pour afficher la courbe des prix
def afficher_courbe_prix(df, selected_carrefour_id, product_column, start_date, end_date):
    # Filtrer les données pour la station Carrefour et les dates sélectionnées
    selected_station = carrefour_stations[carrefour_stations['ID'] == selected_carrefour_id].iloc[0]
    
    # Filtrer les prix pour Carrefour
    carrefour_prices = df[(df['ID'] == selected_carrefour_id) & (df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    # Ajouter les prix des concurrents dans un rayon de 10 km
    competitor_prices = other_stations.copy()
    competitor_prices['Distance_km'] = competitor_prices.apply(
        lambda row: geodesic((selected_station['Latitude'] / 100000, selected_station['Longitude'] / 100000),
                            (row['Latitude'] / 100000, row['Longitude'] / 100000)).kilometers,
        axis=1
    )
    competitor_prices = competitor_prices[competitor_prices['Distance_km'] <= radius_km]
    
    competitors = competitor_prices['ID'].unique()
    competitors_prices = df[(df['ID'].isin(competitors)) & (df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot Carrefour price curve
    plt.plot(carrefour_prices['Date'], carrefour_prices[product_column], label='Carrefour', color='blue', marker='o')
    
    # Plot competitors' price curves
    for competitor_id in competitors:
        competitor_data = competitors_prices[competitors_prices['ID'] == competitor_id]
        plt.plot(competitor_data['Date'], competitor_data[product_column], label=competitor_data['Enseignes'].iloc[0])
    
    plt.title(f"Évolution des prix du {product_column} pour Carrefour et ses concurrents")
    plt.xlabel('Date')
    plt.ylabel('Prix (€)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # Afficher la courbe
    st.pyplot(plt)

# Sélection de la station Carrefour
selected_carrefour_id = st.selectbox(
    "Sélectionnez une station Carrefour",
    carrefour_stations['ID'].unique(),
    format_func=lambda x: carrefour_stations[carrefour_stations['ID'] == x]['Adresse'].iloc[0]
)

# Affichage de la carte
afficher_carte(selected_carrefour_id)

# Sélection de la plage de dates pour l'évolution des prix
# Convertir les dates disponibles en objets datetime.date
available_dates_ = [datetime.datetime.strptime(date, "%Y-%m-%d").date() for date in available_dates]

# Sélection de la plage de dates pour l'évolution des prix
start_date = st.date_input("Sélectionnez la date de début", min_value=min(available_dates_), max_value=max(available_dates_))
end_date = st.date_input("Sélectionnez la date de fin", min_value=start_date, max_value=max(available_dates_))

# Vérification (optionnelle) que les dates sont bien au format datetime.date
st.write(f"Date de début sélectionnée: {start_date}")
st.write(f"Date de fin sélectionnée: {end_date}")

# Affichage du tableau de comparaison des prix pour chaque carburant
for product in list_product_:
    afficher_tableau_comparaison(df, selected_carrefour_id, product)

# Affichage des courbes de prix
for product in list_product_:
    afficher_courbe_prix(df, selected_carrefour_id, product, start_date, end_date)




