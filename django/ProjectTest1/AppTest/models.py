from django.db import models


class ChiffreAffaire(models.Model):
    date = models.CharField(max_length=7, unique=True)  # Garder le format "YYYY-MM"
    chiffre_affaire = models.DecimalField(max_digits=10, decimal_places=2, null=True)  # Permet `null`

    def __str__(self):
        return f"{self.date} - {self.chiffre_affaire if self.chiffre_affaire else 'Aucune donnée'}"
