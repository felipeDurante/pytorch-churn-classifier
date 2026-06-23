import pandas as pd

from projeto_ml.data import REQUIRED_COLUMNS, load_telco_churn


def test_load_telco_churn_returns_dataframe(tmp_path):
    # create a minimal csv matching required columns
    csv = tmp_path / "mini.csv"
    df = pd.DataFrame(
        {
            "CustomerID": [1],
            "Tenure Months": [1],
            "Monthly Charges": [10.0],
            "Total Charges": [10.0],
            "Contract": ["Month-to-month"],
            "Churn Label": ["No"],
            "Churn Value": [0],
        }
    )
    df.to_csv(csv, sep=";", index=False)

    loaded = load_telco_churn(csv)
    assert isinstance(loaded, pd.DataFrame)
    assert REQUIRED_COLUMNS.issubset(set(loaded.columns))
