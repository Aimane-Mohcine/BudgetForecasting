import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TrimestreComponentComponent } from './trimestre-component.component';

describe('TrimestreComponentComponent', () => {
  let component: TrimestreComponentComponent;
  let fixture: ComponentFixture<TrimestreComponentComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TrimestreComponentComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(TrimestreComponentComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
