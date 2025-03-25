# forecasting/forecasting_methods.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime

#ARIMA ET SARIMA
from pmdarima import auto_arima
# Facbook propeht 
from prophet import Prophet


def prophet_forecast(params):
    """
    Fonction similaire à arima_forecast, 
    mais utilisant Prophet (Facebook Prophet) à la place de pmdarima.
    
    On considère un format d'entrée:
      {
        "historique_data": [
          {"date": "2023-01", "revenue": 12345.67},
          {"date": "2023-02", "revenue": 23456.78},
          {"date": "2024-01", "revenue": null},
          ...
        ],
        "freq": "M"  # ou "Q", ou "Y"
      }

    Retourne une liste de dicts:
      [
        {"date": "2023-01", "revenue": 12345.67},
        {"date": "2023-02", "revenue": 23456.78},
        {"date": "2024-01", "revenue": ...},  # prédiction
        ...
      ]
    """

    historique_data = params.get("historique_data", [])
    freq = params.get("freq")

    # Vérification
    if not historique_data or not freq:
        return []

    # On construit un DataFrame
    df = pd.DataFrame(historique_data)  # colonnes: ['date', 'revenue']

    # 1) Convertir la colonne 'date' en un format date (YYYY-MM-DD) compréhensible pour Prophet
    def parse_date(date_str: str, frequency: str) -> str:
        """
        Convertit:
          - freq='M': '2023-01'  => '2023-01-01'
          - freq='Q': '2023-T1' => '2023-01-01', T2 => '2023-04-01', etc.
          - freq='Y': '2023'    => '2023-01-01'
        """
        if frequency == 'M':
            # '2023-01' => '2023-01-01'
            return date_str + "-01"
        elif frequency == 'Q':
            # '2023-T1' => '2023-01-01'
            # '2023-T2' => '2023-04-01'
            # '2023-T3' => '2023-07-01'
            # '2023-T4' => '2023-10-01'
            year, q_str = date_str.split('-')
            quarter = int(q_str.replace('T',''))
            # Calcul du mois de départ (1, 4, 7, 10)
            month_start = 1 + (quarter - 1) * 3
            # Construire une date: f"{year}-{month_start:02d}-01"
            return f"{year}-{month_start:02d}-01"
        elif frequency == 'Y':
            # '2023' => '2023-01-01'
            return date_str + "-01-01"
        else:
            raise ValueError(f"Fréquence non gérée: {frequency}")

    # Appliquer la fonction parse_date pour créer la colonne 'ds'
    df['ds'] = df['date'].apply(lambda d: parse_date(d, freq))

    # Convertir en datetime
    df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d')

    # 2) Séparer l'historique (y non-nulls) du futur (y = None)
    df_histo = df.dropna(subset=['revenue']).copy()
    df_future = df[df['revenue'].isna()].copy()

    # Prophet requiert des colonnes 'ds' (datetime) et 'y' (valeur numérique)
    # Donc on renomme la colonne 'revenue' -> 'y' UNIQUEMENT pour l'historique
    df_histo = df_histo.rename(columns={'revenue': 'y'})

    # Vérif: s'il n'y a pas de points historiques, on ne peut pas entraîner Prophet
    if df_histo.empty:
        # On renvoie la liste brute (dont certains points sont None)
        return df[['date', 'revenue']].to_dict(orient='records')

    # 3) Entraîner le modèle Prophet
      # Choix de la saisonnalité en fonction de freq
    if freq == 'M' or freq == 'Q':
        # On autorise la saisonnalité annuelle
        yearly_s = True
    else:
        # Annuel ou autre
        yearly_s = False

    # Instanciation du modèle
    model = Prophet(
        yearly_seasonality=yearly_s,
        weekly_seasonality=False,
        daily_seasonality=False
    )


    model.fit(df_histo[['ds', 'y']])  # on entraîne sur l'historique

    # 4) Effectuer la prédiction
    # On va prédire sur L'ENSEMBLE (historique + futur), 
    # c'est-à-dire qu'on fournit à Prophet toutes les dates (ds).
    # Prophet renvoie yhat pour toutes ces dates.
    # => il suffit de construire un DataFrame 'future' avec la colonne ds
    future_df = df[['ds']].drop_duplicates().sort_values('ds')

    forecast = model.predict(future_df)

    # Dans 'forecast', on aura des colonnes ['ds', 'yhat', 'yhat_lower', 'yhat_upper', ...]
    # On va juste récupérer 'ds' et 'yhat'.

    forecast_df = forecast[['ds', 'yhat']].copy()

    # 5) Fusionner forecast_df avec df (car on veut revenir au format date original)
    # d'abord, on fusionne sur la colonne 'ds' (datetime)
    merged = pd.merge(
        df, 
        forecast_df, 
        on='ds', 
        how='left'
    )

    # 'merged' contient maintenant: ['date', 'revenue', 'ds', 'yhat']
    # Pour les lignes futures (revenue=None), 'yhat' est la prédiction
    # Pour les lignes historiques, 'revenue' est la valeur réelle
    # On veut en sortie: 'date', 'revenue' (ou la prédiction)...

    # Donc pour tout ce qui est None dans 'revenue', on met 'yhat'
    merged['final_revenue'] = merged.apply(
        lambda row: row['yhat'] if pd.isna(row['revenue']) else row['revenue'],
        axis=1
    )

    # Convertir final_revenue en float
    merged['final_revenue'] = merged['final_revenue'].astype(float)

    # 6) Reconstruire la liste finale
    # On tri par la colonne 'date' si nécessaire (en s'assurant d'un tri "naturel")
    # ou on peut rester en l'état si c'est déjà trié.
    # Au besoin, trier par 'ds' ou par un tri custom (ex. pour Q).
    # Pour simplifier, on part du principe que 'date' est déjà triable
    # ou on se base sur 'ds' qui est chronologique.
    merged.sort_values('ds', inplace=True)

    # On fabrique la liste [ {"date":..., "revenue":...}, ... ]
    final_list = merged[['date', 'final_revenue']].rename(
        columns={'final_revenue': 'revenue'}
    ).to_dict(orient='records')

    return final_list





def arima_forecast(params):

    historique_data=params["historique_data"]
    freq=params["freq"]
    

    # Si on n'a pas de données ou pas de fréquence, on renvoie une liste vide
    if not historique_data or not freq:
        return []

    # Conversion de la liste de dict en DataFrame
    # historique_data est de la forme: 
    # [{"date": "...", "revenue": float ou None}, ...]
    df = pd.DataFrame(historique_data)

    

    
    df.sort_index(inplace=True)

    print("-------------------------------------------------------------")
    print(df)
    print("-------------------------------------------------------------")
    # Séparer l'historique (valeurs non nulles) du futur (valeurs nulles)
    df_histo = df.dropna(subset=['revenue']).copy()  # historique
    df_future = df[df['revenue'].isna()].copy()      # futur à prédire

    
 

    # Séries temporelles pour l'entraînement
    y = df_histo['revenue']
  
    # Entraînement du modèle auto_arima
    # (pour gérer la saisonnalité mensuelle, vous pouvez mettre seasonal=True, m=12)
    model = auto_arima(
        y,
        start_p=1, start_q=1,
        max_p=5, max_q=5,
        d=None,          # détecte automatiquement l'ordre d'intégration
        seasonal=False,  # à ajuster si nécessaire
        stepwise=True,
        trace=False
    )


    # Prédiction sur le nombre de périodes futures
    n_periods = len(df_future)


    # Prédiction sur le nombre de périodes futures
    n_periods = len(df_future)
    if n_periods > 0:
        forecast = model.predict(n_periods=n_periods)
        # Remplir df_future
        df_future['revenue'] = forecast


    # Concaténer historique + futur
    df_combined = pd.concat([df_histo, df_future]).sort_index()



   
   # Remettre l'index à zéro si vous voulez ignorer "period"
    df_combined = df_combined.reset_index(drop=True)

    # forcer revenue en float ou np.float64
    df_combined['revenue'] = df_combined['revenue'].astype(float)

    # Convertir en liste de dict
    final_list = df_combined[['date', 'revenue']].to_dict(orient='records')


   
    return final_list
def sarima_forecast(params):
    """
    Fonction similaire à arima_forecast, mais paramétrée pour un modèle SARIMA
    (donc saisonnier). On choisit un m différent selon freq:
      - freq='M' => m=12  (cycle annuel)
      - freq='Q' => m=4   (4 trimestres/an)
      - freq='Y' => m=1   (pas de saisonnalité plus fine qu'un an)
    """

 

    historique_data = params["historique_data"]
    freq = params["freq"]

    # Si on n'a pas de données ou pas de fréquence, on renvoie une liste vide
    if not historique_data or not freq:
        return []

    # Conversion de la liste de dict en DataFrame
    df = pd.DataFrame(historique_data)

   

    # Choix de m en fonction de freq pour la saisonnalité
    if freq == 'M':
        m = 12
    elif freq == 'Q':
        m = 4
    elif freq == 'Y':
        m = 1  # Pas de cycle plus petit qu'une année
    else:
        raise ValueError(f"Fréquence non gérée : {freq}")

    # On crée une colonne 'period' pour un index temporel ordonné
   
    df.sort_index(inplace=True)
    print("-------------------------------------------------------------")
    print(df)
    print("-------------------------------------------------------------")
    # Séparer l'historique (valeurs non nulles) du futur (valeurs nulles)
    df_histo = df.dropna(subset=['revenue']).copy()
    df_future = df[df['revenue'].isna()].copy()

    # Extraire la série temporelle
    y = df_histo['revenue']

    # Entraînement SARIMA via auto_arima (saisonnalité activée)
    # On passe seasonal=True et on définit m
    model = auto_arima(
        y,
        start_p=1, start_q=1,
        max_p=5, max_q=5,
        d=None,            # auto-détection de l'ordre d'intégration
        seasonal=True,     # Active la saisonnalité
        m=m,               # périodicité saisonnière
        stepwise=True,
        trace=False
    )

    # Prédiction sur le nombre de périodes futures
    n_periods = len(df_future)
    if n_periods > 0:
        forecast = model.predict(n_periods=n_periods)
        df_future['revenue'] = forecast

    # Concaténer historique + futur
    df_combined = pd.concat([df_histo, df_future]).sort_index()

    # Remettre l'index à zéro (on ignore la colonne 'period')
    df_combined = df_combined.reset_index(drop=True)

    # Convertir la colonne 'revenue' en float
    df_combined['revenue'] = df_combined['revenue'].astype(float)

    # Convertir en liste de dict (['date','revenue'])
    final_list = df_combined[['date', 'revenue']].to_dict(orient='records')

    return final_list


def regression_lineaire(params):

    dataset=params["historique_data"]
    print("---------regression_lineaire commence -------")

 
    # 📌 1️⃣ Conversion des dates en indices séquentiels
    unique_dates = sorted(set(entry["date"] for entry in dataset))  # Trier les dates
    date_to_index = {date: i + 1 for i, date in enumerate(unique_dates)}  # Index séquentiel

    # 📌 2️⃣ Appliquer les index séquentiels aux entrées
    for entry in dataset:
        entry["index"] = date_to_index[entry["date"]]

    # 📌 3️⃣ Séparer les données en `train_set` et `forecast_set`
    train_set = [entry for entry in dataset if entry.get("revenue") is not None]
    forecast_set = [entry for entry in dataset if entry.get("revenue") is None]


    print("train_set",train_set)
    print("------------------------------------------------------------")

    print("forecast_set",forecast_set)

    # 📌 5️⃣ Préparer les données pour la régression linéaire
    X_train = np.array([entry["index"] for entry in train_set]).reshape(-1, 1)
    y_train = np.array([entry["revenue"] for entry in train_set])
    # 📌 6️⃣ Entraîner le modèle de régression linéaire
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("------------------------------------------------------------")
    print("✅ Pente (coef_) :", model.coef_[0])
    print("✅ Intercept (intercept_) :", model.intercept_)
    print("------------------------------------------------------------")

  # 📌 7️⃣ Prédire les valeurs des dates inconnues
    X_forecast = np.array([entry["index"] for entry in forecast_set]).reshape(-1, 1)
    y_forecast = model.predict(X_forecast)
    # 📌 8️⃣ Remplacer les valeurs nulles de `forecast_set` par les prévisions
    for i, entry in enumerate(forecast_set):
        entry["revenue"] = round(y_forecast[i], 2)  # Arrondi à 2 décimales

    # 📌 9️⃣ Fusionner et trier les données par date
    complete_data = sorted(train_set + forecast_set, key=lambda x: x["date"])
    print("---------> la fin : ",complete_data)
    return complete_data






def lstm_forecast(previous_revenue):
    """ Forecasting avec une multiplication par 4 """
    return previous_revenue * 4 if previous_revenue is not None else None

# Fonction pour choisir la bonne méthode de forecasting
def get_forecast_function(model):
    """ Retourne la fonction de forecasting correspondant au modèle choisi """
    forecast_methods = {
        "regression_lineaire": regression_lineaire,
        "arima": arima_forecast,
        "sarima": sarima_forecast,
        "prophet":prophet_forecast

    }
    return forecast_methods.get(model)  # Par défaut : simple_forecast
