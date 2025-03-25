from django.urls import path
from .views import simple_message,forecast,reset_and_import_csv,revenus_par_mois
from .views import revenus_par_mois, revenus_par_trimestre, revenus_par_annee

urlpatterns = [
    path('message/', simple_message),
    path('forecast/', forecast),
    path('import/', reset_and_import_csv),
    path('getAll/', revenus_par_mois),
    path('revenus/mois/', revenus_par_mois, name='revenus_par_mois'),  
    path('revenus/trimestre/', revenus_par_trimestre, name='revenus_par_trimestre'),  
    path('revenus/annee/', revenus_par_annee, name='revenus_par_annee'),  

]
