import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==== Paramètres ====
date_debut = "2020-01"
annees = 5
nb_mois_valide = annees * 12
nb_mois_null = 12

# ==== Génération des données ====
np.random.seed(42)
temps = np.arange(nb_mois_valide)

# ✅ Chiffre d'affaires simulé :
# - Tendance linéaire (pas exponentielle)
# - Pas de saisonnalité
# - Bruit aléatoire

base = 50000
trend = base + temps * 800  # croissance linéaire simple
noise = np.random.normal(scale=4000, size=nb_mois_valide)  # bruit raisonnable

valeurs_connues = trend + noise

# ==== Générer les dates ====
dates_connues = pd.date_range(start=date_debut, periods=nb_mois_valide, freq='MS').strftime('%Y-%m')
date_fin_connue = pd.to_datetime(dates_connues[-1]) + pd.DateOffset(months=1)
dates_null = pd.date_range(start=date_fin_connue, periods=nb_mois_null, freq='MS').strftime('%Y-%m')

# ==== DataFrame ====
df_valide = pd.DataFrame({'date': dates_connues, 'value': valeurs_connues})
df_null = pd.DataFrame({'date': dates_null, 'value': [None]*nb_mois_null})
df_final = pd.concat([df_valide, df_null], ignore_index=True)

# ==== Sauvegarde CSV ====
df_final.to_csv("data.csv", index=False)

# ==== Visualisation ====
plt.figure(figsize=(12, 5))
plt.plot(df_final['date'], df_final['value'], marker='o', label='Chiffre d\'affaires')
plt.axvline(x=df_final['date'][nb_mois_valide-1], color='red', linestyle='--', label='Début prévisions')
plt.xticks(rotation=45)
plt.title("Simulation de Chiffre d'Affaires Mensuel (Sans Saisonnalité)")
plt.xlabel("Date")
plt.ylabel("Valeur (en Dhs)")
plt.legend()
plt.tight_layout()
plt.show()
