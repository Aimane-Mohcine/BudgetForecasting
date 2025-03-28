import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { RouterModule } from '@angular/router';
import {HttpClient, HttpClientModule} from "@angular/common/http";

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet,RouterModule,HttpClientModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'test';



  constructor(private http: HttpClient) {}

  /**
   * ✅ Fonction pour mettre à jour les données via une requête API.
   */
  updateData() {
    console.log("🔄 Mise à jour des données en cours...");

    this.http.get('http://localhost:8000/api/import/')
      .subscribe(response => {
        console.log("✅ Mise à jour réussie :", response);
        location.reload();  // ✅ Rafraîchir la page après l'alerte
        alert("🔄 Données mises à jour avec succès !");
      }, error => {
        console.error("❌ Erreur lors de la mise à jour :", error);
        alert("❌ Erreur lors de la mise à jour des données !");
      });
  }
}
