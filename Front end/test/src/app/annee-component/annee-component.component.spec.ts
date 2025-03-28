import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AnneeComponentComponent } from './annee-component.component';

describe('AnneeComponentComponent', () => {
  let component: AnneeComponentComponent;
  let fixture: ComponentFixture<AnneeComponentComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AnneeComponentComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(AnneeComponentComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
