"""Script para gerar dataset processado em data/processed/ a partir de raw."""
from pathlib import Path

from projeto_ml.data import save_processed


def main() -> None:
    project_root = Path.cwd()
    raw = project_root / "data" / "raw" / "Telco_customer_churn(Telco_Churn).csv"
    out = project_root / "data" / "processed" / "telco_processed.csv"
    saved = save_processed(raw, out)
    print(f"Saved processed dataset to: {saved}")


if __name__ == "__main__":
    main()
