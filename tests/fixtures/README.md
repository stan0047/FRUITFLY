# Test fixtures — NOT REAL BIOLOGICAL DATA

**Everything in this directory is a test fixture. None of it is anatomy.**

These files are hand-written so that the ingestion pipeline can be tested
without a real dataset, on any machine, with no download. They are not derived
from FlyWire, from any connectome reconstruction, or from any publication. The
neuron identifiers, regions and types are arbitrary strings chosen to exercise
the importer's code paths.

Two consequences the tests rely on:

* Fixtures are imported with `DataSource.TEST_FIXTURE`, which is structurally
  incapable of reporting `is_biological=True`. A test fixture cannot be
  mistaken for real data even if a caller asks for it.
* Reports and figures built from a fixture carry the label
  `TEST FIXTURE - NOT REAL BIOLOGICAL DATA`.

| File | Purpose |
| --- | --- |
| `sample_edges.csv` | happy path: `pre_id`, `post_id`, `synapse_count`, region and type columns |
| `sample_edges_duplicates.csv` | repeated `(source, target)` pairs, a self-loop, a null endpoint, a negative weight |
| `sample_nodes.csv` | neuron metadata table, deliberately missing one neuron used in the edge file |
| `sample_edges_ambiguous.csv` | several plausible column names per role, so inference must refuse |
| `sample_edges_no_weight.csv` | no weight column at all, so a flagged placeholder weight is produced |
| `sample_edges_malformed.csv` | a non-numeric weight value |
| `sample_edges_empty.csv` | header only, no rows |

Real biological data, when it arrives, goes in `data/raw/` and is never
committed. See `../../data/README.md`.
