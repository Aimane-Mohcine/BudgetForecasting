import { Component, ViewChild, ElementRef, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { FormsModule } from "@angular/forms";
import Chart from 'chart.js/auto';

@Component({
  selector: 'app-tester-donnees',
  standalone: true,
  imports: [CommonModule, HttpClientModule, FormsModule],
  templateUrl: './tester-donnees.component.html',
  styleUrl: './tester-donnees.component.css'
})
export class TesterDonneesComponent implements AfterViewInit {

  @ViewChild('chartCanvas') chartCanvas!: ElementRef<HTMLCanvasElement>;
  revenueChart: any;

  selectedModel: string = 'regression_lineaire';
  isLoading = false;
  showPrecision = false;
  precision = 91.5;
  meilleurModele: string | null = null;
  allPrecisions: { [key: string]: number } | null = null;


  typeAffichage: 'mois' | 'trimestre' | 'annee' = 'mois';

  // ✅ Deux tableaux : un pour tout (graphe), un filtré (tableau)
  allSalesData: { date: string, revenue: number | null }[] = [];
  salesData: { date: string, revenue: number }[] = [];

  constructor(private http: HttpClient) {}

  ngAfterViewInit() {
    this.loadData('mois'); // données initiales
  }

  onModelChange() {
    this.showPrecision = false;
    this.updateChart();
  }

  changerAffichage(type: 'mois' | 'trimestre' | 'annee') {
    this.typeAffichage = type;
    this.showPrecision = false;
    this.loadData(type);
  }
  testerModel() {
    this.isLoading = true;
    this.showPrecision = false;
    this.meilleurModele = null;
    this.allPrecisions = null;

    const payload = {
      model: this.selectedModel,
      type: this.typeAffichage,
      data: this.salesData
    };

    this.http.post<any>('http://localhost:8000/api/tester-modele/', payload).subscribe(response => {
      this.precision = response.precision;
      this.showPrecision = true;
      this.isLoading = false;

      // Cas "best_fit" → récupérer le meilleur modèle et les précisions complètes
      if (this.selectedModel === "best_fit") {
        this.meilleurModele = response.meilleur_modele;
        this.allPrecisions = response.tous_les_resultats;
      } else {
        this.meilleurModele = this.selectedModel;
      }
    }, error => {
      console.error("❌ Erreur lors du test :", error);
      this.isLoading = false;
    });
  }




  loadData(type: 'mois' | 'trimestre' | 'annee') {
    let url = '';

    if (type === 'mois') url = 'http://localhost:8000/api/revenus/mois/';
    else if (type === 'trimestre') url = 'http://localhost:8000/api/revenus/trimestre/';
    else if (type === 'annee') url = 'http://localhost:8000/api/revenus/annee/';

    this.http.get<any>(url).subscribe(data => {
      let rawData: { date: string, revenue: number | null }[] = [];

      if (type === 'mois') {
        rawData = data.chiffres_affaires.map((item: any) => ({
          date: item.date,
          revenue: item.chiffre_affaire !== null ? Number(item.chiffre_affaire) : null
        }));
      } else {
        rawData = Object.entries(data).map(([date, revenue]: [string, any]) => ({
          date,
          revenue: revenue !== null ? Number(revenue) : null
        }));
      }

      this.allSalesData = rawData;
      this.salesData = rawData.filter(item => item.revenue !== null) as { date: string, revenue: number }[];

      if (this.revenueChart) {
        this.updateChart();
      } else {
        this.createChart();
      }
    }, error => {
      console.error("❌ Erreur lors du chargement des données :", error);
    });
  }

  createChart() {
    const ctx = this.chartCanvas.nativeElement.getContext('2d');
    if (!ctx) return;

    this.revenueChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: this.allSalesData.map(d => d.date),
        datasets: [
          {
            label: 'Chiffre d\'Affaires (€)',
            data: this.allSalesData.map(d => d.revenue),
            borderColor: 'blue',
            backgroundColor: 'blue',
            borderWidth: 2
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
      this.revenueChart.data.labels = this.allSalesData.map(d => d.date);
      this.revenueChart.data.datasets[0].data = this.allSalesData.map(d => d.revenue);
      this.revenueChart.update();
    }
  }
}
