import { Component, ViewChild, ElementRef, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { FormsModule } from "@angular/forms";
import Chart from 'chart.js/auto';

@Component({
  selector: 'app-mois-component',
  standalone: true,
  imports: [CommonModule, HttpClientModule, FormsModule],
  templateUrl: './mois-component.component.html',
  styleUrl: './mois-component.component.css'
})
export class MoisComponentComponent implements AfterViewInit {

  @ViewChild('chartCanvas') chartCanvas!: ElementRef<HTMLCanvasElement>;  // Récupération du canvas pour le graphique
  revenueChart: any;  // Instance du graphique Chart.js

  salesData: any[] = [];  // Tableau contenant l'historique des ventes

  selectedModel: string = 'regression_lineaire';  // Modèle sélectionné par défaut
  isLoading = false;  // Indicateur de chargement

  constructor(private http: HttpClient) {}

  ngAfterViewInit() {
    this.loadSalesData();
    setTimeout(() => {
      this.createChart();
    }, 100); // Petit délai pour s'assurer que le DOM est prêt
  }

  /**
   * Charge les données des ventes depuis l'API Django.
   */
  loadSalesData() {
    this.isLoading = true;

    this.http.get<{ chiffres_affaires: any[] }>('http://localhost:8000/api/revenus/mois/')
      .subscribe(response => {
        console.log("📥 Données reçues :", response);

        if (response && response.chiffres_affaires) {
          this.salesData = response.chiffres_affaires.map(historique_data => ({
            date: historique_data.date,
            revenue: historique_data.chiffre_affaire,
            forecasted: historique_data.chiffre_affaire === null
          }));
          this.isLoading = false;
          this.updateChart();
        } else {
          console.error("❌ Erreur : Données manquantes dans la réponse API");
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
      model: this.selectedModel,
      freq:"M"
    };

    console.log("📤 Envoi des données :", payload);

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

  updateChart() {
    if (this.revenueChart) {
      const { realData, forecastData } = this.prepareChartData();

      this.revenueChart.data.labels = this.salesData.map(historique_data => historique_data.date);
      this.revenueChart.data.datasets[0].data = realData;
      this.revenueChart.data.datasets[1].data = forecastData;
      this.revenueChart.update();
    }
  }

  prepareChartData() {
    let realData: (number | null)[] = [];
    let forecastData: (number | null)[] = [];
    let lastRealValue: number | null = null;

    this.salesData.forEach((historique_data, index) => {
      if (!historique_data.forecasted) {
        realData.push(historique_data.revenue);
        forecastData.push(null);
        lastRealValue = historique_data.revenue;
      } else {
        if (forecastData.length > 0 && forecastData[forecastData.length - 1] === null && lastRealValue !== null) {
          forecastData[index - 1] = lastRealValue;
        }
        realData.push(null);
        forecastData.push(historique_data.revenue);
      }
    });

    return { realData, forecastData };
  }

}
