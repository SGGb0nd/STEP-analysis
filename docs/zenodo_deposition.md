# Zenodo deposition

The deposit is built from `docs/zenodo_manifest.tsv`. Public source datasets
marked `link_only` remain external references and are not duplicated. The
archives contain analysis-ready data, generated intermediates, figures, metric
tables, and other direct analysis outputs. Source code, notebooks, reports, and
repository documentation remain in the linked GitHub repository.

Full AnnData objects produced by comparison methods are not redistributed.
The deposit retains STEP outputs and the compact assignments, metrics, and
figures needed to inspect the reported comparisons.

Validate all ready manifest entries without writing archives:

```bash
python scripts/deposition/build_zenodo_archives.py --validate-only
```

Build one deterministic archive per manifest tier:

```bash
python scripts/deposition/build_zenodo_archives.py
```

The final build refuses to proceed while the manifest contains deferred rows.
Resolve or remove those rows before packaging. Partial archives require the
explicit `--allow-deferred` flag and are not final deposition artifacts.

The command writes `STEP_analysis_raw.tar.zst`,
`STEP_analysis_intermediate.tar.zst`, `STEP_analysis_output.tar.zst`, an
archive inventory, `archive_summary.tsv`, and SHA-256 checksums under
`artifacts/zenodo/`.
The inventory records the manifest checksum; the uploader rejects archives
built from an older manifest.

Archive staging excludes operating-system metadata, notebook caches, logs,
source documents, and redundant source ZIP files. The source data directories
are not modified.

Create or update a Zenodo draft after setting `ZENODO_TOKEN`:

```bash
uv run python scripts/deposition/upload_zenodo_draft.py \
  --archives-dir artifacts/zenodo \
  --metadata-json docs/zenodo_metadata.json
```

The uploader never publishes the record. After reviewing the draft, validate
the complete remote file set and publish it with:

```bash
uv run python scripts/deposition/finalize_zenodo_draft.py \
  --deposit-id DEPOSIT_ID \
  --state-json artifacts/zenodo/zenodo_draft_state.json
```
