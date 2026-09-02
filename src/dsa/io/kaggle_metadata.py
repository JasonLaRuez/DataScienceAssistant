"""Step 1: fetching a Kaggle dataset's own description, for display alongside a load.

Uses kagglehub's own internal client (it's what kagglehub itself is built on) rather
than a separate Kaggle client library, so this needs no new dependency and authenticates
exactly the way dsa.io.kaggle.load_kaggle already does.

Column-level metadata is deliberately not attempted here: Kaggle's optional per-column
"Data Card" field is empty for most real datasets (checked against several before
deciding this), so a feature built on it would rarely deliver what "feature meanings"
implies. The dataset page link this module surfaces is the fallback for that.
"""

from __future__ import annotations

from dataclasses import dataclass

from dsa.session import Session

STEP = 1


def _fetch_dataset_info(owner_slug: str, dataset_slug: str):
    """The one network call this module makes. A bare function so tests can monkeypatch
    it directly, the same way dsa.io.kaggle.fetch is stubbed. Imported lazily: loading
    the toolkit should not require network code."""
    import kagglesdk.datasets.types.dataset_api_service as dataset_api_service
    from kagglehub.clients import build_kaggle_client

    client = build_kaggle_client()
    request = dataset_api_service.ApiGetDatasetMetadataRequest()
    request.owner_slug = owner_slug
    request.dataset_slug = dataset_slug
    response = client.datasets.dataset_api_client.get_dataset_metadata(request)
    return response.info


@dataclass(frozen=True)
class DatasetMetadata:
    """A Kaggle dataset's own title/subtitle/description -- not a substitute for the
    actual page, which is why the link is included alongside it."""

    slug: str
    title: str
    subtitle: str
    description: str

    @property
    def url(self) -> str:
        # Kaggle's API has its own 'url' field, but it was empty on every dataset
        # checked while building this -- built from the slug instead, the one thing
        # guaranteed present.
        return f"https://www.kaggle.com/datasets/{self.slug}"

    def describe(self) -> str:
        lines = [self.title]
        if self.subtitle:
            lines.append(self.subtitle)
        lines.append(self.url)
        if self.description:
            lines.append("")
            lines.append(self.description)
        return "\n".join(lines)

    def _repr_markdown_(self) -> str:
        """Jupyter calls this automatically for the last expression in a cell -- no
        IPython import needed anywhere, the same duck-typed display protocol
        matplotlib Figures already rely on."""
        parts = [f"### [{self.title}]({self.url})"]
        if self.subtitle:
            parts.append(f"*{self.subtitle}*")
        if self.description:
            parts.append(self.description)
        return "\n\n".join(parts)

    def __repr__(self) -> str:
        return self.describe()


def fetch_dataset_metadata(slug: str) -> DatasetMetadata:
    """Fetch a Kaggle dataset's title/subtitle/description directly from Kaggle."""
    owner_slug, _, dataset_slug = slug.partition("/")
    info = _fetch_dataset_info(owner_slug, dataset_slug)
    return DatasetMetadata(
        slug=slug,
        title=info.title or slug,
        subtitle=info.subtitle,
        description=info.description,
    )


def describe_dataset(session: Session) -> DatasetMetadata:
    """Fetch and log this session's Kaggle dataset's description (step 1 continued).

    Requires the session to have been loaded via :func:`dsa.load_kaggle` -- raises a
    clear error otherwise, since a locally loaded file has no Kaggle page to describe.
    """
    if session.source is None or not session.source.startswith("kaggle:"):
        raise ValueError(
            "describe_dataset only applies to a session loaded via dsa.load_kaggle "
            f"(session.source is {session.source!r})"
        )
    slug = session.source.removeprefix("kaggle:")

    with session.log.record(STEP, "load.describe_dataset", {"slug": slug}):
        metadata = fetch_dataset_metadata(slug)

    return metadata
