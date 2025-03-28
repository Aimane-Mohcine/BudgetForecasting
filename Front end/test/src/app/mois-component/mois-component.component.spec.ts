import { ComponentFixture, TestBed } from '@angular/core/testing';

import { MoisComponentComponent } from './mois-component.component';

describe('MoisComponentComponent', () => {
  let component: MoisComponentComponent;
  let fixture: ComponentFixture<MoisComponentComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [MoisComponentComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(MoisComponentComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
