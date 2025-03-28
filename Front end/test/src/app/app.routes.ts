import { Routes } from '@angular/router';
import {TestComponent} from "./test/test.component";
import {MoisComponentComponent} from "./mois-component/mois-component.component";
import {TrimestreComponentComponent} from "./trimestre-component/trimestre-component.component";
import {AnneeComponentComponent} from "./annee-component/annee-component.component";
import {TesterDonneesComponent} from "./tester-donnees/tester-donnees.component";

export const routes: Routes = [

  { path: 'tester-donnees', component: TesterDonneesComponent },
  { path: '', component: MoisComponentComponent },
  { path: 'test', component: TestComponent },
  { path: 'mois', component: MoisComponentComponent },
  { path: 'trimestre', component: TrimestreComponentComponent },
  { path: 'annee', component: AnneeComponentComponent },
];
