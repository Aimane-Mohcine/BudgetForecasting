import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==== Paramètres ====
date_debut = "2021-01"
annee = 4
nb_mois_valide = annee * 12  # 48 mois de données historiques
nb_mois_null = 12            # 12 mois à prédire (futurs)

# ==== Génération des données ====
np.random.seed(42)
temps = np.arange(nb_mois_valide)  # 0, 1, 2, ... 47

# Capacité logistique (pour la colonne 'cap' de Prophet)
capacite = 120000

# Croissance logistique vers la capacité
# Milieu (point d'inflexion) vers le mois 24 pour montrer une saturation progressive
logistic_growth = capacite / (1 + np.exp(-0.05 * (temps - 24)))

# Saisonnalité mensuelle (sinusoïde de période 12 mois)
seasonality = 10000 * np.sin(2 * np.pi * temps / 12)

# Bruit aléatoire
noise = np.random.normal(scale=3000, size=nb_mois_valide)

# Valeur finale
valeurs_connues = logistic_growth + seasonality + noise

# ==== Générer les dates ====
dates_connues = pd.date_range(start=date_debut, periods=nb_mois_valide, freq='MS')
date_fin_connue = dates_connues[-1] + pd.DateOffset(months=1)
dates_futures = pd.date_range(start=date_fin_connue, periods=nb_mois_null, freq='MS')

# ==== Construction du DataFrame pour Prophet ====
# Partie "historique" (avec valeurs)
df_valide = pd.DataFrame({
    'ds': dates_connues,
    'y': valeurs_connues,
    'cap': [capacite] * nb_mois_valide  # même capacité à chaque date
})

# Partie "futures" (sans valeurs, y=None)
df_null = pd.DataFrame({
    'ds': dates_futures,
    'y': [None]*nb_mois_null,
    'cap': [capacite] * nb_mois_null
})

# Concaténer les deux parties
df_final = pd.concat([df_valide, df_null], ignore_index=True)

# Concaténer les deux parties
df_final = pd.concat([df_valide, df_null], ignore_index=True)

# Formater les dates comme "2023-01"
df_final['ds'] = df_final['ds'].dt.strftime('%Y-%m')


# ==== Sauvegarde CSV ====
df_final.to_csv("data.csv", index=False)

# ==== Visualisation ====
plt.figure(figsize=(12, 5))
plt.plot(df_valide['ds'], df_valide['y'], marker='o', label='Chiffre d\'affaires (historique)')
plt.axvline(x=df_valide['ds'].iloc[-1], color='red', linestyle='--', label='Début prévisions')
plt.xticks(rotation=45)
plt.title("Simulation de Chiffre d'Affaires Mensuel (Adapté à Prophet)")
plt.xlabel("Date")
plt.ylabel("Valeur (en Dhs)")
plt.legend()
plt.tight_layout()
plt.show()
