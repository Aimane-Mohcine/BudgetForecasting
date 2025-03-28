import { Component, ViewChild, ElementRef, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { FormsModule } from "@angular/forms";
import Chart from 'chart.js/auto';

@Component({
  selector: 'app-annee-component',
  standalone: true,
  imports: [CommonModule, HttpClientModule, FormsModule],
  templateUrl: './annee-component.component.html',
  styleUrl: './annee-component.component.css'
})
export class AnneeComponentComponent implements AfterViewInit {

  @ViewChild('chartCanvas') chartCanvas!: ElementRef<HTMLCanvasElement>;
  revenueChart: any;

  salesData: { annee: string, revenue: number | null, forecasted: boolean }[] = [];
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
   * ✅ Charge les revenus annuels depuis l'API Django.
   * ✅ Remplace les `null` pour identifier les revenus inconnus.
   */
  loadSalesData() {
    this.isLoading = true;

    this.http.get<{ [key: string]: number | null }>('http://localhost:8000/api/revenus/annee/')
      .subscribe(response => {
        console.log("📥 Données reçues :", response);

        if (response) {
          this.salesData = Object.keys(response).map(annee => ({
            annee: annee,
            revenue: response[annee], // ✅ Garde les null sans remplacer
            forecasted: response[annee] === null // ✅ Marque les valeurs inconnues
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
   * ✅ Réinitialise les prévisions en mettant les valeurs inconnues à `null`.
   */
  resetForecasting() {
    this.salesData.forEach(anneeData => {
      if (anneeData.forecasted) {
        anneeData.revenue = null;
        anneeData.forecasted = false;
      }
    });
    this.updateChart();
  }

  /**
   * ✅ Envoie les données au backend pour obtenir une prévision.
   */
  getForecasting() {
    this.isLoading = true;
    this.resetForecasting();

    const payload = {
      historique_data: this.salesData.map(anneeData => ({
        date: anneeData.annee,
        revenue: anneeData.revenue

      })),
      model: this.selectedModel,
      freq:"Y"
    };

    console.log("📤 Envoi des données :", payload);

    this.http.post('http://localhost:8000/api/forecast/', payload)
      .subscribe((response: any) => {
        console.log("📥 Réponse reçue :", response);

        this.salesData.forEach((anneeData, index) => {
          if (anneeData.revenue === null) {
            anneeData.revenue = response.forecast[index];
            anneeData.forecasted = true;
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
   * ✅ Prépare les données pour le graphique avec une transition fluide.
   */
  prepareChartData() {
    let realData: (number | null)[] = [];
    let forecastData: (number | null)[] = [];
    let lastRealValue: number | null = null;

    this.salesData.forEach((anneeData, index) => {
      if (!anneeData.forecasted) {
        realData.push(anneeData.revenue);
        forecastData.push(null);
        lastRealValue = anneeData.revenue;
      } else {
        if (forecastData.length > 0 && forecastData[forecastData.length - 1] === null && lastRealValue !== null) {
          forecastData[index - 1] = lastRealValue;
        }
        realData.push(null);
        forecastData.push(anneeData.revenue);
      }
    });

    return { realData, forecastData };
  }

  /**
   * ✅ Met à jour le graphique avec transition entre réel et forecast.
   */
  updateChart() {
    if (this.revenueChart) {
      const { realData, forecastData } = this.prepareChartData();

      this.revenueChart.data.labels = this.salesData.map(anneeData => anneeData.annee);
      this.revenueChart.data.datasets[0].data = realData;
      this.revenueChart.data.datasets[1].data = forecastData;
      this.revenueChart.update();
    }
  }

  /**
   * ✅ Crée le graphique annuel sans modifier le type.
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

    const { realData, forecastData } = this.prepareChartData();

    this.revenueChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: this.salesData.map(anneeData => anneeData.annee),
        datasets: [
          {
            label: 'Revenu réel',
            data: realData,
            borderColor: 'blue',
            backgroundColor: 'blue',
            borderWidth: 2
          },
          {
            label: 'Prévision (forecasting)',
            data: forecastData,
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
}
