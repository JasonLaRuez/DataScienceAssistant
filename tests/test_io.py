"""Tests for step 1: credential detection, table discovery, and loading.

Every test here is offline. Kaggle's network call is the one thing stubbed out; the file
discovery, gate, and logging behaviour around it is exercised for real.
"""

from __future__ import annotations

import pandas as pd
import pytest

import dsa
from dsa.io.kaggle import require_credentials
from dsa.io.readers import find_tables, table_format

KAGGLE_ENV_VARS = ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_CONFIG_DIR")


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """No ambient Kaggle credentials, and a config dir that is guaranteed empty.

    Without pointing KAGGLE_CONFIG_DIR at a temp directory these tests would pass or fail
    depending on whether the developer running them happens to have a real kaggle.json.
    """
    for var in KAGGLE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path / "no-such-config"))


@pytest.fixture
def session(tmp_path):
    return dsa.new_session(project_root=tmp_path)


@pytest.fixture
def dataset(tmp_path):
    """A stand-in for a downloaded Kaggle dataset directory holding a single table."""
    directory = tmp_path / "download"
    directory.mkdir()
    pd.DataFrame({"age": [22, 38, 26], "fare": [7.25, 71.3, 7.9], "survived": [0, 1, 1]}).to_csv(
        directory / "train.csv", index=False
    )
    return directory


# --- readers -------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, expected",
    [
        ("train.csv", ".csv"),
        ("train.CSV", ".csv"),
        ("train.csv.gz", ".csv"),      # compression is stripped before dispatch
        ("data.parquet", ".parquet"),
        ("sheet.xlsx", ".xlsx"),
        ("notes.md", None),
        ("archive.zip", None),         # compression alone is not a table
    ],
)
def test_table_format_ignores_compression(name, expected):
    assert table_format(name) == expected


def test_find_tables_recurses_and_skips_archive_noise(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "__MACOSX").mkdir()
    (tmp_path / ".hidden").mkdir()
    for path in ("a.csv", "nested/b.parquet", "__MACOSX/c.csv", ".hidden/d.csv", "readme.md"):
        (tmp_path / path).write_text("x", encoding="utf-8")

    found = [p.relative_to(tmp_path).as_posix() for p in find_tables(tmp_path)]
    assert found == ["a.csv", "nested/b.parquet"]


def test_read_table_roundtrips_csv_and_parquet(tmp_path):
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    frame.to_csv(tmp_path / "t.csv", index=False)
    frame.to_parquet(tmp_path / "t.parquet")

    pd.testing.assert_frame_equal(dsa.read_table(tmp_path / "t.csv"), frame)
    pd.testing.assert_frame_equal(dsa.read_table(tmp_path / "t.parquet"), frame)


def test_read_table_rejects_an_unknown_format(tmp_path):
    (tmp_path / "notes.md").write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="don't know how to read"):
        dsa.read_table(tmp_path / "notes.md")


# --- credentials ---------------------------------------------------------------------

def test_credential_precedence_matches_kagglehub(clean_env, monkeypatch, tmp_path):
    assert dsa.credential_source() is None

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "kaggle.json").write_text('{"username": "u", "key": "k"}', encoding="utf-8")
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(config_dir))
    assert "credentials file" in dsa.credential_source()

    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.setenv("KAGGLE_KEY", "k")
    assert dsa.credential_source() == "KAGGLE_USERNAME / KAGGLE_KEY environment variables"

    monkeypatch.setenv("KAGGLE_API_TOKEN", "KGAT_x")
    assert dsa.credential_source() == "KAGGLE_API_TOKEN environment variable"


def test_missing_credentials_explain_all_three_mechanisms(clean_env):
    with pytest.raises(dsa.CredentialsMissing) as excinfo:
        require_credentials()
    message = str(excinfo.value)
    assert "KAGGLE_API_TOKEN" in message
    assert "KAGGLE_USERNAME" in message
    assert "kaggle.json" in message
    assert "restart the kernel" in message.lower()


def test_credential_value_never_reaches_the_log(clean_env, monkeypatch, session, dataset):
    """Only the *mechanism* is recorded, never the secret itself."""
    monkeypatch.setenv("KAGGLE_API_TOKEN", "KGAT_supersecret")
    monkeypatch.setattr("dsa.io.kaggle.fetch", lambda slug, cache_dir: dataset)

    dsa.load_kaggle(session, "someone/dataset")

    serialised = "".join(entry.to_json() for entry in session.log.entries)
    assert "KGAT_supersecret" not in serialised
    assert "KAGGLE_API_TOKEN environment variable" in serialised  # the mechanism is logged


# --- loading -------------------------------------------------------------------------

@pytest.fixture
def authed(clean_env, monkeypatch):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "KGAT_test")


def test_load_kaggle_populates_the_session(authed, monkeypatch, session, dataset):
    monkeypatch.setattr("dsa.io.kaggle.fetch", lambda slug, cache_dir: dataset)

    dsa.load_kaggle(session, "someone/titanic")

    assert session.source == "kaggle:someone/titanic"
    assert session.raw.shape == (3, 3)
    # The working frame starts as a copy of raw: no repairs have been approved yet.
    pd.testing.assert_frame_equal(session.df, session.raw)

    ops = [e.op for e in session.log.entries]
    assert "load.fetch" in ops and "load.read" in ops
    read_entry = next(e for e in session.log.entries if e.op == "load.read")
    assert read_entry.output_shape == (3, 3)


def test_loading_opens_but_does_not_force_the_target_gate(authed, monkeypatch, session, dataset):
    """Step 1 completes; naming a target needs a look at the data first."""
    monkeypatch.setattr("dsa.io.kaggle.fetch", lambda slug, cache_dir: dataset)
    dsa.load_kaggle(session, "someone/titanic")

    gate = session.gates["target"]
    assert not gate.answered
    assert gate.options == ("age", "fare", "survived")
    assert "survived" in gate.context  # the column preview helps make the choice

    dsa.decide(session, "target", "survived")
    assert dsa.require(session, "target") == "survived"


def test_multiple_tables_open_a_gate_and_the_retry_succeeds(authed, monkeypatch, session, tmp_path):
    directory = tmp_path / "multi"
    directory.mkdir()
    pd.DataFrame({"a": [1], "y": [0]}).to_csv(directory / "train.csv", index=False)
    pd.DataFrame({"a": [2]}).to_csv(directory / "test.csv", index=False)
    monkeypatch.setattr("dsa.io.kaggle.fetch", lambda slug, cache_dir: directory)

    with pytest.raises(dsa.GateRequired, match="Which one should be loaded"):
        dsa.load_kaggle(session, "someone/multi")

    dsa.decide(session, "source_file", "train.csv")
    dsa.load_kaggle(session, "someone/multi")  # cached; re-running the cell just works
    assert list(session.raw.columns) == ["a", "y"]


def test_naming_a_missing_file_lists_what_is_available(authed, monkeypatch, session, dataset):
    monkeypatch.setattr("dsa.io.kaggle.fetch", lambda slug, cache_dir: dataset)
    with pytest.raises(FileNotFoundError, match="train.csv"):
        dsa.load_kaggle(session, "someone/titanic", file="nope.csv")


def test_loading_a_second_dataset_discards_stale_repairs(authed, monkeypatch, session, dataset):
    """Repairs were approved for the previous frame; carrying them over would be wrong."""
    monkeypatch.setattr("dsa.io.kaggle.fetch", lambda slug, cache_dir: dataset)
    dsa.load_kaggle(session, "someone/titanic")
    session.repairs = [("drop_fare", lambda df: df.drop(columns=["fare"]))]

    dsa.load_kaggle(session, "someone/titanic")
    assert session.repairs == []


# --- dataset description ---------------------------------------------------------------

class _FakeDatasetInfo:
    def __init__(self, title: str, subtitle: str, description: str):
        self.title = title
        self.subtitle = subtitle
        self.description = description


def _stub_dataset_info(monkeypatch, title="Titanic Dataset", subtitle="Survival prediction", description="Some description."):
    monkeypatch.setattr(
        "dsa.io.kaggle_metadata._fetch_dataset_info",
        lambda owner_slug, dataset_slug: _FakeDatasetInfo(title, subtitle, description),
    )


def test_fetch_dataset_metadata_builds_the_url_from_the_slug(monkeypatch):
    _stub_dataset_info(monkeypatch)
    metadata = dsa.fetch_dataset_metadata("yasserh/titanic-dataset")

    assert metadata.url == "https://www.kaggle.com/datasets/yasserh/titanic-dataset"
    assert metadata.title == "Titanic Dataset"


def test_dataset_metadata_markdown_includes_title_link_and_description(monkeypatch):
    _stub_dataset_info(monkeypatch, description="Some description.")
    metadata = dsa.fetch_dataset_metadata("yasserh/titanic-dataset")

    markdown = metadata._repr_markdown_()
    assert "Titanic Dataset" in markdown
    assert "https://www.kaggle.com/datasets/yasserh/titanic-dataset" in markdown
    assert "Some description." in markdown


def test_describe_dataset_requires_a_kaggle_loaded_session(session, dataset):
    dsa.load_file(session, dataset / "train.csv")
    with pytest.raises(ValueError, match="load_kaggle"):
        dsa.describe_dataset(session)


def test_describe_dataset_fetches_and_logs(authed, monkeypatch, session, dataset):
    monkeypatch.setattr("dsa.io.kaggle.fetch", lambda slug, cache_dir: dataset)
    dsa.load_kaggle(session, "someone/titanic")
    _stub_dataset_info(monkeypatch, title="Titanic")

    metadata = dsa.describe_dataset(session)

    assert metadata.title == "Titanic"
    entry = next(e for e in session.log.entries if e.op == "load.describe_dataset")
    assert entry.params == {"slug": "someone/titanic"}
