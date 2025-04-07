from sklearn.linear_model import LinearRegression

import numpy as np

import random
import time  
from sklearn.metrics import mean_absolute_error
from pmdarima import auto_arima
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

def tester_regression_lineaire() -> float:
      # Génère une précision aléatoire entre 85.00 et 95.00
    precision = round(random.uniform(85, 95), 2)
    print("✅ Précision simulée :", precision, "%")
     # 💤 Pause simulée (ex: 1.5 secondes)
    time.sleep(3)
    return precision


def evaluate_regression_lineaire(type_affichage, data, nb_periodes_test=1):

    print("▶️ Évaluation modèle - Régression Linéaire")
    print("📆 Type affichage :", type_affichage)
    m=12
    t=4
    a=1
    # Définir combien de lignes selon le type_affichage
    type_to_multiplier = {
        "mois": m*1,
        "trimestre": t*1,
        "annee": a*2
    }
    
    if type_affichage not in type_to_multiplier:
        raise ValueError("Type d'affichage non reconnu. Utilisez 'mois', 'trimestre' ou 'annee'.")
    
    horizon_test = nb_periodes_test * type_to_multiplier[type_affichage]
    print(f"🧮 Nombre de lignes test à prévoir : {horizon_test}")

    # Trier les dates et indexer
    unique_dates = sorted(set(entry["date"] for entry in data))
    date_to_index = {date: i for i, date in enumerate(unique_dates)}
    
    for entry in data:
        entry["index"] = date_to_index[entry["date"]]

    # Trier le dataset
    dataset_sorted = sorted(data, key=lambda x: x["index"])
    complete_entries = [e for e in dataset_sorted if e["revenue"] is not None]

    if len(complete_entries) <= horizon_test:
        print("⚠️ Pas assez de données pour créer un test set")
        return 0

    train_set = complete_entries[:-horizon_test]
    test_set = complete_entries[-horizon_test:]
    print("------------------------------------")
    print("traing sets :")
    for item in train_set:
        print(f"  ➤ {item['date']} : {item['revenue']} €")
    print("------------------------------------")
    print("test sets :")
    for item in test_set:
        print(f"  ➤ {item['date']} : {item['revenue']} €")
    print("------------------------------------")
    X_train = np.array([e["index"] for e in train_set]).reshape(-1, 1)
    y_train = np.array([e["revenue"] for e in train_set])
    
    model = LinearRegression()
    model.fit(X_train, y_train)

    X_test = np.array([e["index"] for e in test_set]).reshape(-1, 1)
    y_test = np.array([e["revenue"] for e in test_set])
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    precision = 1 - mae / (np.mean(y_test) + 1e-8)
    precision = round(precision * 100, 2)

    print("✅ Précision :", precision, "%")
    return precision



def evaluate_arima(type_affichage, data, nb_periodes_test=1):
    print("▶️ Évaluation modèle - ARIMA")
    print("📆 Type affichage :", type_affichage)

    m, t, a = 12, 4, 1
    type_to_multiplier = {
        "mois": m * 1,
        "trimestre": t * 1,
        "annee": a * 2
    }

    if type_affichage not in type_to_multiplier:
        raise ValueError("Type d'affichage non reconnu. Utilisez 'mois', 'trimestre' ou 'annee'.")

    horizon_test = nb_periodes_test * type_to_multiplier[type_affichage]
    print(f"🧮 Nombre de lignes test à prévoir : {horizon_test}")

    # 📦 Conversion en DataFrame
    df = pd.DataFrame(data)
    df = df.sort_values("date").reset_index(drop=True)

    # 📌 Historique uniquement
    df_histo = df[df["revenue"].notna()].copy()

    if len(df_histo) <= horizon_test:
        print("⚠️ Pas assez de données pour créer un test set")
        return 0

    # 📤 Séparation train/test
    df_train = df_histo.iloc[:-horizon_test]
    df_test = df_histo.iloc[-horizon_test:]

    print("------------------------------------")
    print("training sets :")
    for i, row in df_train.iterrows():
        print(f"  ➤ {row['date']} : {row['revenue']} €")
    print("------------------------------------")
    print("test sets :")
    for i, row in df_test.iterrows():
        print(f"  ➤ {row['date']} : {row['revenue']} €")
    print("------------------------------------")

    y_train = df_train["revenue"]
    y_test = df_test["revenue"]

    # 🧠 Entraînement ARIMA
    model = auto_arima(
        y_train,
        start_p=1, start_q=1,
        max_p=5, max_q=5,
        d=None,
        seasonal=False,
        stepwise=True,
        trace=False
    )

    y_pred = model.predict(n_periods=horizon_test)

    # 🎯 MAPE & précision
    mape_val = mean_absolute_percentage_error(y_test, y_pred) * 100
    precision = round(100 - mape_val, 2)

    print(f"✅ Précision ARIMA : {precision:.2f}% (MAPE = {mape_val:.2f}%)")
    return precision




def evaluate_sarima(type_affichage, data, nb_periodes_test=1):
    print("▶️ Évaluation modèle - SARIMA (saisonnier)")
    print("📆 Type affichage :", type_affichage)

    # Choix de la fréquence saisonnière pour m
    if type_affichage == 'mois':
        m = 12
        
    elif type_affichage == 'trimestre':
        m = 4
       
    elif type_affichage == 'annee':
        m = 1
        
    else:
        raise ValueError("Type d'affichage non reconnu. Utilisez 'mois', 'trimestre' ou 'annee'.")

    type_to_multiplier = {
        "mois": 12,
        "trimestre": 4,
        "annee": 2  
    }

    horizon_test = nb_periodes_test * type_to_multiplier[type_affichage]
    print(f"🧮 Nombre de lignes test à prévoir : {horizon_test}")

    # 📦 Conversion en DataFrame
    df = pd.DataFrame(data)
    df = df.sort_values("date").reset_index(drop=True)

    df_histo = df[df["revenue"].notna()].copy()

    if len(df_histo) <= horizon_test:
        print("⚠️ Pas assez de données pour créer un test set")
        return 0

    # 🔁 Séparation train/test
    df_train = df_histo.iloc[:-horizon_test]
    df_test = df_histo.iloc[-horizon_test:]

    print("------------------------------------")
    print("training sets :")
    for i, row in df_train.iterrows():
        print(f"  ➤ {row['date']} : {row['revenue']} €")
    print("------------------------------------")
    print("test sets :")
    for i, row in df_test.iterrows():
        print(f"  ➤ {row['date']} : {row['revenue']} €")
    print("------------------------------------")

    y_train = df_train['revenue']
    y_test = df_test['revenue']

    # 🧠 Modèle SARIMA avec saisonnalité activée
    model = auto_arima(
        y_train,
        start_p=1, start_q=1,
        max_p=5, max_q=5,
        d=None,
        seasonal=True,
        m=m,
        stepwise=True,
        trace=False
    )

    y_pred = model.predict(n_periods=horizon_test)

    # 🎯 Évaluation
    mape_val = mean_absolute_percentage_error(y_test, y_pred) * 100
    precision = round(100 - mape_val, 2)

    print(f"✅ Précision SARIMA : {precision:.2f}% (MAPE = {mape_val:.2f}%)")
    return precision



def evaluate_prophet(type_affichage, data, nb_periodes_test=1):
    print("▶️ Évaluation modèle - Prophet")
    print("📆 Type affichage :", type_affichage)

    # Choix de la fréquence (et saisonnalité)
    if type_affichage == 'mois':
        freq = 'M'
        m = 12
    elif type_affichage == 'trimestre':
        freq = 'Q'
        m = 4
    elif type_affichage == 'annee':
        freq = 'Y'
        m = 2
    else:
        raise ValueError("Type d'affichage non reconnu")

    horizon_test = nb_periodes_test * m
    print(f"🧮 Nombre de lignes test à prévoir : {horizon_test}")

    df = pd.DataFrame(data)

    # Convertir 'date' en 'ds' utilisable par Prophet
    def parse_date(date_str, frequency):
        if frequency == 'M':
            return date_str + "-01"
        elif frequency == 'Q':
            year, quarter = date_str.split('-')
            month_start = 1 + (int(quarter.replace('T','')) - 1) * 3
            return f"{year}-{month_start:02d}-01"
        elif frequency == 'Y':
            return date_str + "-01-01"

    df['ds'] = df['date'].apply(lambda d: parse_date(d, freq))
    df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d')

    df_histo = df[df['revenue'].notna()].copy().sort_values('ds')
    
    if len(df_histo) <= horizon_test:
        print("⚠️ Pas assez de données pour créer un test set")
        return 0

    df_train = df_histo.iloc[:-horizon_test]
    df_test = df_histo.iloc[-horizon_test:]

    print("------------------------------------")
    print("training sets :")
    for _, row in df_train.iterrows():
        print(f"  ➤ {row['date']} : {row['revenue']} €")
    print("------------------------------------")
    print("test sets :")
    for _, row in df_test.iterrows():
        print(f"  ➤ {row['date']} : {row['revenue']} €")
    print("------------------------------------")

    # Prophet attend ds + y
    df_train = df_train.rename(columns={'revenue': 'y'})

    model = Prophet(
        yearly_seasonality=(freq in ['M', 'Q']),
        weekly_seasonality=False,
        daily_seasonality=False
    )

    model.fit(df_train[['ds', 'y']])

    future = df_test[['ds']]
    forecast = model.predict(future)

    y_true = df_test['revenue'].values
    y_pred = forecast['yhat'].values

    mape_val = mean_absolute_percentage_error(y_true, y_pred) * 100
    precision = round(100 - mape_val, 2)

    print(f"✅ Précision Prophet : {precision:.2f}% (MAPE = {mape_val:.2f}%)")
    return precision




def evaluate_holt_winters(type_affichage, data, nb_periodes_test=1):
    print("▶️ Évaluation modèle - Holt-Winters")
    print("📆 Type affichage :", type_affichage)

    warnings.filterwarnings("ignore", category=UserWarning)

    # Déterminer fréquence et saisonnalité
    if type_affichage == "mois":
        seasonal_periods = 12
        freq = "M"
    elif type_affichage == "trimestre":
        seasonal_periods = 4
        freq = "Q"
    elif type_affichage == "annee":
        seasonal_periods = None
        freq = "Y"
    else:
        raise ValueError("Type d'affichage non reconnu")

    type_to_multiplier = {
        "mois": 12,
        "trimestre": 4,
        "annee": 2  # au moins 2 ans
    }

    horizon_test = nb_periodes_test * type_to_multiplier[type_affichage]
    print(f"🧮 Nombre de lignes test à prévoir : {horizon_test}")

    df = pd.DataFrame(data).sort_values("date").reset_index(drop=True)
    df_histo = df[df["revenue"].notna()].copy()

    if len(df_histo) <= horizon_test:
        print("⚠️ Pas assez de données pour créer un test set")
        return 0

    # Découper train/test
    df_train = df_histo.iloc[:-horizon_test]
    df_test = df_histo.iloc[-horizon_test:]

    print("------------------------------------")
    print("training sets :")
    for _, row in df_train.iterrows():
        print(f"  ➤ {row['date']} : {row['revenue']} €")
    print("------------------------------------")
    print("test sets :")
    for _, row in df_test.iterrows():
        print(f"  ➤ {row['date']} : {row['revenue']} €")
    print("------------------------------------")

    y_train = df_train["revenue"].astype(float)
    y_test = df_test["revenue"].astype(float)

    best_model = None
    best_aic = np.inf
    best_config = ""

    if seasonal_periods:
        for trend in ['add', 'mul']:
            for seasonal in ['add', 'mul']:
                try:
                    model = ExponentialSmoothing(
                        y_train,
                        trend=trend,
                        seasonal=seasonal,
                        seasonal_periods=seasonal_periods
                    ).fit()
                    if model.aic < best_aic:
                        best_model = model
                        best_aic = model.aic
                        best_config = f"trend={trend}, seasonal={seasonal}"
                except:
                    continue
    else:
        for trend in ['add', 'mul']:
            try:
                model = ExponentialSmoothing(
                    y_train,
                    trend=trend,
                    seasonal=None
                ).fit()
                if model.aic < best_aic:
                    best_model = model
                    best_aic = model.aic
                    best_config = f"trend={trend}, seasonal=None"
            except:
                continue

    # Prédiction et évaluation
    if best_model:
        y_pred = best_model.forecast(horizon_test)
        mape_val = mean_absolute_percentage_error(y_test, y_pred) * 100
        precision = round(100 - mape_val, 2)

        print(f"✅ Meilleur modèle : {best_config} (AIC = {best_aic:.2f})")
        print(f"✅ Précision Holt-Winters : {precision:.2f}% (MAPE = {mape_val:.2f}%)")
        return precision
    else:
        print("❌ Aucun modèle Holt-Winters valide trouvé.")
        return 0


def evaluate_best_model(type_affichage, data, nb_periodes_test=1):
    print("🔍 Sélection du meilleur modèle en cours...")

    results = {}

    try:
        results["Regression Linéaire"] = evaluate_regression_lineaire(type_affichage, data.copy(), nb_periodes_test)
    except Exception as e:
        print("⚠️ Erreur Regression Linéaire :", e)
        results["Regression Linéaire"] = 0

    try:
        results["ARIMA"] = evaluate_arima(type_affichage, data.copy(), nb_periodes_test)
    except Exception as e:
        print("⚠️ Erreur ARIMA :", e)
        results["ARIMA"] = 0

    try:
        results["SARIMA"] = evaluate_sarima(type_affichage, data.copy(), nb_periodes_test)
    except Exception as e:
        print("⚠️ Erreur SARIMA :", e)
        results["SARIMA"] = 0

    try:
        results["Prophet"] = evaluate_prophet(type_affichage, data.copy(), nb_periodes_test)
    except Exception as e:
        print("⚠️ Erreur Prophet :", e)
        results["Prophet"] = 0

    try:
        results["Holt-Winters"] = evaluate_holt_winters(type_affichage, data.copy(), nb_periodes_test)
    except Exception as e:
        print("⚠️ Erreur Holt-Winters :", e)
        results["Holt-Winters"] = 0

    # 🔽 Trier par précision décroissante
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

    best_model = next(iter(sorted_results))
    best_precision = sorted_results[best_model]

    print("\n📊 Résultats triés :")
    for model, precision in sorted_results.items():
        print(f"  ➤ {model} : {precision} %")

    print(f"\n🏆 Meilleur modèle : {best_model} ({best_precision} %)")
    return {
        "meilleur_modele": best_model,
        "precision": best_precision,
        "tous_les_resultats": sorted_results
    }



# Fonction pour choisir la bonne méthode 
def get_evaluate_forcing_algorithms(model):
    """ Retourne la fonction de forecasting correspondant au modèle choisi """
    forecast_methods = {
        "regression_lineaire": evaluate_regression_lineaire,
        "arima": evaluate_arima,
        "sarima": evaluate_sarima,
        "prophet":evaluate_prophet,
        "holt-Winters":evaluate_holt_winters,
        "best_fit":evaluate_best_model
        

    }
    return forecast_methods.get(model)  # Par défaut : simple_forecast