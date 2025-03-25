import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==== 🔧 Paramètres ====
date_debut = "2020-01"
annees = 5              # Nombre d'années avec données
nb_mois_valide = annees * 12
nb_mois_null = 12       # Mois à prédire (valeurs manquantes)
nom_fichier = "data.csv"

# ==== 📈 Génération des données ====
np.random.seed(42)  # Pour que le bruit soit reproductible
base = 1000
pas = 200
bruit_max = 500  # max +/- pour le bruit

valeurs_connues = [
    base + i * pas + np.random.randint(-bruit_max, bruit_max)
    for i in range(nb_mois_valide)
]

# ==== 🗓 Génération des dates ====
dates_connues = pd.date_range(start=date_debut, periods=nb_mois_valide, freq='MS').strftime('%Y-%m')
date_fin_connue = pd.to_datetime(dates_connues[-1]) + pd.DateOffset(months=1)
dates_null = pd.date_range(start=date_fin_connue, periods=nb_mois_null, freq='MS').strftime('%Y-%m')

# ==== 🧩 DataFrames ====
df_valide = pd.DataFrame({'date': dates_connues, 'value': valeurs_connues})
df_null = pd.DataFrame({'date': dates_null, 'value': [None] * nb_mois_null})
df_final = pd.concat([df_valide, df_null], ignore_index=True)

# ==== 💾 Sauvegarde ====
df_final.to_csv(nom_fichier, index=False)
print(f"✅ Fichier généré : {nom_fichier}")

# ==== 📊 Visualisation ====
plt.figure(figsize=(12, 5))
plt.plot(df_final['date'], df_final['value'], marker='o', label='Valeurs simulées')
plt.axvline(x=df_final['date'][nb_mois_valide-1], color='red', linestyle='--', label='Début prévisions')
plt.xticks(rotation=45)
plt.title("Série Temporelle Linéaire + Bruit Léger")
plt.xlabel("Date")
plt.ylabel("Valeur")
plt.legend()
plt.tight_layout()
plt.show()
