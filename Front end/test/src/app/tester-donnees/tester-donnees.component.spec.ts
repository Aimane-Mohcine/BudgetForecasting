import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TesterDonneesComponent } from './tester-donnees.component';

describe('TesterDonneesComponent', () => {
  let component: TesterDonneesComponent;
  let fixture: ComponentFixture<TesterDonneesComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TesterDonneesComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(TesterDonneesComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
