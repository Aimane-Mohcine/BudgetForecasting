import { Component, ViewChild, ElementRef, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { FormsModule } from "@angular/forms";
import Chart from 'chart.js/auto';

@Component({
  selector: 'app-test',
  standalone: true,
  imports: [CommonModule, HttpClientModule, FormsModule],
  templateUrl: './test.component.html',
  styleUrl: './test.component.css'
})
export class TestComponent implements AfterViewInit  {
  @ViewChild('chartCanvas') chartCanvas!: ElementRef<HTMLCanvasElement>;  // ✅ Récupération du canvas
  revenueChart: any;
/*
  salesData =
    [

      {
        "date": "2024-02-01",
        "revenue": 6450,
        "forecasted": false
      },
      {
        "date": "2024-03-01",
        "revenue": 6500,
        "forecasted": false
      },
      {
        "date": "2024-04-01",
        "revenue": 6550,
        "forecasted": false
      },
      {
        "date": "2024-05-01",
        "revenue": 6600,
        "forecasted": false
      },
      {
        "date": "2024-06-01",
        "revenue": 6650,
        "forecasted": false
      },
      {
        "date": "2024-07-01",
        "revenue": 6700,
        "forecasted": false
      },
      {
        "date": "2024-08-01",
        "revenue": 6750,
        "forecasted": false
      },
      {
        "date": "2024-09-01",
        "revenue": 6800,
        "forecasted": false
      },
      {
        "date": "2024-10-01",
        "revenue": 6850,
        "forecasted": false
      },
      {
        "date": "2024-11-01",
        "revenue": 6900,
        "forecasted": false
      },
      {
        "date": "2024-12-01",
        "revenue": 6950,
        "forecasted": false
      },
      {
        "date": "2025-01-01",
        "revenue": null,
        "forecasted": true
      },
      {
        "date": "2025-02-01",
        "revenue": null,
        "forecasted": true
      },
      {
        "date": "2025-03-01",
        "revenue": null,
        "forecasted": true
      },
      {
        "date": "2025-04-01",
        "revenue": null,
        "forecasted": true
      },
      {
        "date": "2025-05-01",
        "revenue": null,
        "forecasted": true
      },
      {
        "date": "2025-06-01",
        "revenue": null,
        "forecasted": true
      },
      {
        "date": "2025-07-01",
        "revenue": null,
        "forecasted": true
      },
      {
        "date": "2025-08-01",
        "revenue": null,
        "forecasted": true
      },
      {
        "date": "2025-09-01",
        "revenue": null,
        "forecasted": true
      },
      {
        "date": "2025-10-01",
        "revenue": null,
        "forecasted": true
      },
      {
        "date": "2025-11-01",
        "revenue": null,
        "forecasted": true
      },
      {
        "date": "2025-12-01",
        "revenue": null,
        "forecasted": true
      }
    ];
*/
  salesData: any[] = [];

  selectedModel: string = 'regression_lineaire';
  isLoading = false;

  constructor(private http: HttpClient) {}

  // ✅ Exécuté après l'affichage du composant pour créer le graphique
  ngAfterViewInit() {
    this.loadSalesData();  // ✅ Charger les données dès que le composant s'affiche

    setTimeout(() => {
      this.createChart();
    }, 100); // ✅ Petit délai pour s'assurer que le DOM est bien prêt


  }

  loadSalesData() {
    this.isLoading = true;

    this.http.get<{ chiffres_affaires: any[] }>('http://localhost:8000/api/revenus/mois/')
      .subscribe(response => {
        console.log("📥 Données reçues :", response);

        // ✅ Vérifier si la clé "chiffres_affaires" existe et contient des données
        if (response && response.chiffres_affaires) {
          this.salesData = response.chiffres_affaires.map(historique_data => ({
            date: historique_data.date,
            revenue: historique_data.chiffre_affaire,
            forecasted: historique_data.chiffre_affaire === null  // ✅ Si null, alors forecasted = true
          }));

          this.isLoading = false;
        } else {
          console.error("❌ Erreur : Données manquantes dans la réponse de l'API");
          this.isLoading = false;
        }
      }, error => {
        console.error("❌ Erreur lors du chargement des données :", error);
        this.isLoading = false;
      });
  }

        onModelChange() {
    console.log("🔄 Modèle sélectionné :", this.selectedModel);
    this.resetForecasting();
  }

  resetForecasting() {
    this.salesData.forEach(historique_data => {
      if (historique_data.forecasted) {
        historique_data.revenue = null;
        historique_data.forecasted = false;
      }
    });
    this.updateChart();
  }

  getForecasting() {
    this.isLoading = true;
    this.resetForecasting();

    const payload = {
      historique_data: this.salesData.map(historique_data => ({
        date: historique_data.date,
        revenue: historique_data.revenue
      })),
      model: this.selectedModel
    };

    console.log("📤 Payload envoyé :", payload);

    this.http.post('http://localhost:8000/api/forecast/', payload)
      .subscribe((response: any) => {
        console.log("📥 Réponse reçue :", response);

        this.salesData.forEach((historique_data, index) => {
          if (historique_data.revenue === null) {
            historique_data.revenue = response.forecast[index];
            historique_data.forecasted = true;
          }
        });

        this.isLoading = false;
        this.updateChart();
      }, error => {
        console.error("❌ Erreur lors de la récupération des prévisions :", error);
        this.isLoading = false;
      });
  }






  // ✅ Créer le graphique avec ViewChild
  createChart() {
    if (!this.chartCanvas) {
      console.error("❌ Erreur : canvas introuvable !");
      return;
    }

    const ctx = this.chartCanvas.nativeElement.getContext('2d');
    if (!ctx) {
      console.error("❌ Erreur : impossible d'obtenir le contexte du canvas !");
      return;
    }

    this.revenueChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: this.salesData.map(historique_data => historique_data.date),
        datasets: [
          {
            label: 'Revenu réel',
            data: this.salesData.map(historique_data => historique_data.forecasted ? null : historique_data.revenue),
            borderColor: 'blue',
            backgroundColor: 'blue',
            borderWidth: 2
          },
          {
            label: 'Prévision (forecasting)',
            data: this.salesData.map(historique_data => historique_data.forecasted ? historique_data.revenue : null),
            borderColor: 'red',
            backgroundColor: 'red',
            borderWidth: 2,
            borderDash: [5, 5]
          }
        ]
      },
      options: {
        responsive: true,
        plugins: {
          legend: {
            display: true,
            position: 'top'
          }
        }
      }
    });
  }

  // ✅ Mettre à jour le graphique
  updateChart() {
    if (this.revenueChart) {
      // ✅ Préparer les données réelles et forecast avec transition
      const { realData, forecastData } = this.prepareChartData();

      // ✅ Mise à jour des labels et datasets
      this.revenueChart.data.labels = this.salesData.map(historique_data => historique_data.date);
      this.revenueChart.data.datasets[0].data = realData;
      this.revenueChart.data.datasets[1].data = forecastData;
      this.revenueChart.update();
    }
  }

// ✅ Nouvelle méthode pour générer des données correctement formatées
  prepareChartData() {
    let realData: (number | null)[] = [];
    let forecastData: (number | null)[] = [];
    let lastRealValue: number | null = null;

    this.salesData.forEach((historique_data, index) => {
      if (!historique_data.forecasted) {
        realData.push(historique_data.revenue);
        forecastData.push(null);
        lastRealValue = historique_data.revenue; // ✅ Mise à jour du dernier réel
      } else {
        if (forecastData.length > 0 && forecastData[forecastData.length - 1] === null && lastRealValue !== null) {
          // ✅ Ajouter un point de transition entre réel et forecasting
          forecastData[index - 1] = lastRealValue;
        }
        realData.push(null); // ✅ Éviter d'afficher la valeur en double
        forecastData.push(historique_data.revenue);
      }
    });

    return { realData, forecastData };
  }

}
