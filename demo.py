import sys
from src.generator import generate_lot
from src.pipeline import run_pipeline
from src.constants import ASSUMED_DEFECT_CORRELATION

def main():
    print(f"Running AGNIPARIKSHA demo with ASSUMED_DEFECT_CORRELATION = {ASSUMED_DEFECT_CORRELATION}")
    
    print("Generating lot of 1000 chips...")
    # Seed 42 for reproducibility
    df_long = generate_lot(n_chips=1000, seed=42)
    
    print("Running pipeline...")
    df, model, report = run_pipeline(df_long)
    
    print("\n" + "="*50)
    print("DEMO RESULTS")
    print("="*50)
    print(f"PAT Latent-Defect Recall:           {report.pat_recall_latent*100:.1f}%")
    print(f"Full Pipeline Latent-Defect Recall: {report.full_recall_latent*100:.1f}%")
    print("-" * 50)
    print(f"PAT Precision:                      {report.pat_precision*100:.1f}%")
    print(f"Full Pipeline Precision:            {report.full_precision*100:.1f}%")
    print("-" * 50)
    print(f"PAT F1 Score:                       {report.pat_f1:.3f}")
    print(f"Full Pipeline F1 Score:             {report.full_f1:.3f}")
    print("-" * 50)
    print(f"Conformal Coverage (nominal 95%):   {report.conformal_coverage*100:.1f}%")
    print("="*50)

if __name__ == "__main__":
    main()
