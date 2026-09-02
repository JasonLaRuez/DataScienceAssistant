"""Getting data in: Kaggle fetching and local table reading."""

from dsa.io.kaggle import CredentialsMissing, credential_source, load_kaggle
from dsa.io.kaggle_metadata import DatasetMetadata, describe_dataset, fetch_dataset_metadata
from dsa.io.readers import find_tables, read_table, table_format

__all__ = [
    "CredentialsMissing",
    "credential_source",
    "load_kaggle",
    "describe_dataset",
    "fetch_dataset_metadata",
    "DatasetMetadata",
    "find_tables",
    "read_table",
    "table_format",
]
