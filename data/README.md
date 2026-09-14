# Data directory

BugBrain deliberately does not commit 179 MB of fly-brain data to Git.

Run:

```bash
uv run bugbrain download
uv run bugbrain build
```

The downloader fetches and verifies:

| File | Pinned MD5 |
|---|---|
| `connections_biological.csv.gz` | `2af9b8db6a8dbbc02cd3cc36f2631c19` |
| `neuron_annotations_flywire_v2.1.0.tsv.gz` | `55789fa30a854bd00451af16441a7d3d` |

Source: E. Correig-Fraga, R. Guimerà, and M. Sales-Pardo, *Data and source data for “Structure alone supports efficient visual computation in the Drosophila visual system”*, version 1.0.0, Zenodo, 2026, [doi:10.5281/zenodo.21549559](https://doi.org/10.5281/zenodo.21549559).

The source release is CC BY 4.0. The underlying FlyWire connectivity release is [doi:10.5281/zenodo.10676866](https://doi.org/10.5281/zenodo.10676866). See [NOTICE](../NOTICE) for attribution and project-independence statements.
