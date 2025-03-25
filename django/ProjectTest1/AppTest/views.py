from django.shortcuts import render

from rest_framework.response import Response
from rest_framework.decorators import api_view
from .forecasting_methods import get_forecast_function
from .models import ChiffreAffaire  # Assurez-vous que c'est bien importé

import csv
from datetime import datetime

from django.db.models import Sum
from django.db.models.functions import Substr, Cast


# API qui retourne une liste statique d'articles sans modèle
@api_view(['GET'])
def simple_message(request):
    return Response({"message": "Hello, ceci est une API simple en Django REST Framework !"})


@api_view(['POST'])
def forecast(request):
    

    print("📤 Requête reçue :", request.data)
    # Récupérer les données envoyées par Angular
    params = {
        "historique_data": request.data.get("historique_data", []),
        "model": request.data.get("model"),
        "freq": request.data.get("freq")
    }

    

    
    # Obtenir la fonction correspondant au modèle choisi
    forecast_function = get_forecast_function(params["model"])
    # 4️⃣ Appliquer la régression linéaire pour compléter `forecast_set`
    complete_data = forecast_function(params)
      # ✅ Extraire uniquement la liste des revenus
    revenue_list = [entry["revenue"] for entry in complete_data]

    print("✅ Liste finale des revenus :", revenue_list)

    return Response({"forecast": revenue_list})




#les end point de front end 

@api_view(['GET'])
def reset_and_import_csv(request):
    chemin_fichier = r"C:\Users\hp\Desktop\Projet\Data\data.csv"

    # 1️⃣ **Supprimer toutes les données de la table**
    ChiffreAffaire.objects.all().delete()

    try:
        with open(chemin_fichier, newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Ignorer l'en-tête

            resultats = []
            for row in reader:
                try:
                    date_str = row[0]  # Garder "YYYY-MM" tel quel
                    chiffre_affaire = row[1]  # Lire le chiffre d'affaire (peut être null)

                    # Si chiffre_affaire est vide ou 'NaN', on l'assigne à None
                    if not chiffre_affaire or chiffre_affaire.lower() == 'nan':
                        chiffre_affaire = None
                    else:
                        chiffre_affaire = float(chiffre_affaire)  # Convertir en float

                    # Créer l'objet dans la base de données
                    obj = ChiffreAffaire.objects.create(
                        date=date_str,
                        chiffre_affaire=chiffre_affaire
                    )

                    resultats.append({"date": obj.date, "chiffre_affaire": obj.chiffre_affaire, "status": "Ajouté ✅"})

                except Exception as e:
                    resultats.append({"error": f"❌ Erreur sur la ligne {row}: {e}"})

        return Response({'message': 'Réinitialisation et importation terminées', 'données_insérées': resultats})

    except FileNotFoundError:
        return Response({'error': f"❌ Fichier introuvable : {chemin_fichier}"}, status=404)
    


@api_view(['GET'])
def revenus_par_mois(request):
    data = ChiffreAffaire.objects.all().values("date", "chiffre_affaire")  # Récupérer toutes les lignes
    return Response({"chiffres_affaires": list(data)})

@api_view(['GET'])
def revenus_par_trimestre(request):
    
    trimestres = {}

    # 📌 1️⃣ Regrouper les chiffres d'affaires par trimestre
    for ca in ChiffreAffaire.objects.all():
        annee, mois = map(int, ca.date.split('-'))  # Extraire l'année et le mois
        trimestre = (mois - 1) // 3 + 1  # Calculer le trimestre
        key = f"{annee}-T{trimestre}"  # Clé du trimestre

        # Vérifier si chiffre_affaire est None et éviter les erreurs
        chiffre = ca.chiffre_affaire if ca.chiffre_affaire is not None else 0

        # Ajouter au dictionnaire, en évitant d'additionner `None`
        if key not in trimestres:
            trimestres[key] = chiffre  # Initialisation
        else:
            trimestres[key] += chiffre  # Addition des revenus

    # 📌 2️⃣ Remplacer les valeurs `0` par `None`
    for key in trimestres:
        if trimestres[key] == 0:
            trimestres[key] = None

    return Response(trimestres)

@api_view(['GET'])
def revenus_par_annee(request):
    """ Calculer les revenus par année """
    revenus = (
        ChiffreAffaire.objects
        .annotate(annee=Substr('date', 1, 4))  # Remplacement de Substring par Substr
        .values('annee')
        .annotate(total=Sum('chiffre_affaire'))
        .order_by('annee')
    )

    return Response({r['annee']: r['total'] for r in revenus})