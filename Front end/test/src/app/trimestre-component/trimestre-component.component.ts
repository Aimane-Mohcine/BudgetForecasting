import { Component, ViewChild, ElementRef, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { FormsModule } from "@angular/forms";
import Chart from 'chart.js/auto';

@Component({
  selector: 'app-trimestre-component',
  standalone: true,
  imports: [CommonModule, HttpClientModule, FormsModule],
  templateUrl: './trimestre-component.component.html',
  styleUrl: './trimestre-component.component.css'
})
export class TrimestreComponentComponent implements AfterViewInit {

  @ViewChild('chartCanvas') chartCanvas!: ElementRef<HTMLCanvasElement>;
  revenueChart: any;

  salesData: { trimestre: string, revenue: number | null, forecasted: boolean }[] = [];
  selectedModel: string = 'regression_lineaire';
  isLoading = false;

  constructor(private http: HttpClient) {}

  ngAfterViewInit() {
    this.loadSalesData();
    setTimeout(() => {
      this.createChart();
    }, 100);
  }

  onModelChange() {
    console.log("🔄 Modèle sélectionné :", this.selectedModel);
    this.resetForecasting();
  }
  /**
   * Charge les revenus trimestriels depuis l'API Django.
   * Remplace les `0` par `null` pour identifier les revenus inconnus.
   */
  loadSalesData() {
    this.isLoading = true;

    this.http.get<{ [key: string]: number }>('http://localhost:8000/api/revenus/trimestre/')
      .subscribe(response => {
        console.log("📥 Données reçues :", response);

        if (response) {
          this.salesData = Object.keys(response).map(trimestre => ({
            trimestre: trimestre,
            revenue:  response[trimestre], // ✅ Remplace 0 par null
            forecasted: response[trimestre] === null // ✅ Marque ces valeurs comme prévues
          }));

          console.log("📊 Données transformées pour le tableau :", this.salesData);
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

  /**
   * Réinitialise les prévisions en mettant les valeurs inconnues à `null`.
   */
  resetForecasting() {
    this.salesData.forEach(trimestreData => {
      if (trimestreData.forecasted) {
        trimestreData.revenue = null;
        trimestreData.forecasted = false;
      }
    });
    this.updateChart();
  }

  /**
   * Envoie les données au backend pour obtenir une prévision.
   */
  getForecasting() {
    this.isLoading = true;
    this.resetForecasting();

    const payload = {
      historique_data: this.salesData.map(trimestreData => ({
        date: trimestreData.trimestre,
        revenue: trimestreData.revenue

      })),
      model: this.selectedModel,
      freq:"Q"
    };

    console.log("📤 Envoi des données :", payload);

    this.http.post('http://localhost:8000/api/forecast/', payload)
      .subscribe((response: any) => {
        console.log("📥 Réponse reçue :", response);

        this.salesData.forEach((trimestreData, index) => {
          if (trimestreData.revenue === null) {  // ✅ Vérifie null au lieu de 0
            trimestreData.revenue = response.forecast[index];
            trimestreData.forecasted = true;
          }
        });

        this.isLoading = false;
        this.updateChart();
      }, error => {
        console.error("❌ Erreur lors de la récupération des prévisions :", error);
        this.isLoading = false;
      });
  }

  /**
   * ✅ NE CHANGE PAS LE GRAPHIQUE, il reste exactement le même.
   */
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
      type: 'line', // ✅ Le type du graphique reste inchangé
      data: {
        labels: this.salesData.map(trimestreData => trimestreData.trimestre),
        datasets: [
          {
            label: 'Revenu réel',
            data: this.salesData.map(trimestreData => trimestreData.forecasted ? null : trimestreData.revenue),
            borderColor: 'blue',
            backgroundColor: 'blue',
            borderWidth: 2
          },
          {
            label: 'Prévision (forecasting)',
            data: this.salesData.map(trimestreData => trimestreData.forecasted ? trimestreData.revenue : null),
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

  /**
   * ✅ Prépare les données pour le graphique avec une transition fluide.
   */
  prepareChartData() {
    let realData: (number | null)[] = [];
    let forecastData: (number | null)[] = [];
    let lastRealValue: number | null = null;

    this.salesData.forEach((trimestreData, index) => {
      if (!trimestreData.forecasted) {
        realData.push(trimestreData.revenue);
        forecastData.push(null);
        lastRealValue = trimestreData.revenue; // ✅ Dernière valeur réelle
      } else {
        if (forecastData.length > 0 && forecastData[forecastData.length - 1] === null && lastRealValue !== null) {
          // ✅ Ajoute un point de transition entre réel et forecasting
          forecastData[index - 1] = lastRealValue;
        }
        realData.push(null); // ✅ Évite d'afficher la valeur en double
        forecastData.push(trimestreData.revenue);
      }
    });

    return { realData, forecastData };
  }

  /**
   * ✅ Met à jour le graphique avec transition entre les valeurs réelles et forecast.
   */
  updateChart() {
    if (this.revenueChart) {
      const { realData, forecastData } = this.prepareChartData();

      this.revenueChart.data.labels = this.salesData.map(trimestreData => trimestreData.trimestre);
      this.revenueChart.data.datasets[0].data = realData;
      this.revenueChart.data.datasets[1].data = forecastData;
      this.revenueChart.update();
    }
  }



}
