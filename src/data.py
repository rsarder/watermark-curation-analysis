
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def load_ag_news(validation_fraction=0.10, seed=42):
    """Load AG News and create a reproducible stratified split."""
    dataset = load_dataset("fancyzhx/ag_news")

    train_records = dataset["train"].to_pandas()
    test_records = dataset["test"].to_pandas()

    train_df, val_df = train_test_split(
        train_records,
        test_size=validation_fraction,
        stratify=train_records["label"],
        random_state=seed,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_records.reset_index(drop=True),
    )
