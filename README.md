# Graph-based Class C exploration

Explore colored molecular cycles and figure-eights using human D2 (CHEMBL217) and D3 (CHEMBL234) Ki measurements from the ChEMBL CSVs in `data/raw/`.

```bash
pip install numpy pandas scipy scikit-learn matplotlib torch transformers
```

Run from the repository root, in order:

```bash
python src/01_prepare_dataset.py
python src/02_extract_activations.py
python src/03_pca_whiten.py
python src/04_build_knn_graph.py
python src/05_color_knn_edges.py
python src/06_find_colored_cycles.py
python src/07_find_figure_eights.py
python src/08_plot_figure_eights.py
python src/09_export_twist_rankings.py
```

- **Dataset:** 4,867 matched molecules; median pKi per target. Each target is independently mean-centered and scaled within −50 to 50, producing `d2_d3_m50.csv`. `--no-normalize` produces true pKi in `d2_d3.csv`.
- **Geometry:** last-token activations, layer 18, 50 PCA dimensions, regularized whitening, and 50 nearest neighbors.
- **Edges:** admit D2/red and D3/blue independently when `value(B) - value(A) > 0.05 * abs(value(A))`. Change this with `--min-increase-percent`.
- **Cycles:** enumerate simple 3–4-edge cycles with one red segment and one blue segment, restricted to possible figure-eight participants.
- **Pairs:** two cycles sharing exactly one molecule; save up to 100 per hub, prioritizing lobe rotation. `holonomy.py` fits local ridge frames (`n_loc=256`, ridge 10). Invalid or orientation-reversing frame loops are excluded from this rotation ranking.
- **Figures:** hub counts and galleries with six examples, including distinct cycles, longest saved pairs, and distinct hubs. Counts describe saved representatives.
- **CSV rankings:** Step 09 writes `top100_twist_hubs.csv` and `top100_twist_cycles.csv` beside the figure-eight results. Hubs rank by their largest lobe angle, then the other lobe angle. Cycle ranks cover all Step 06 cycles with a holonomy-valid figure-eight partner, retaining one representative per molecule set. Exports include indices, ChEMBL IDs, SMILES, and fitted binding values (normalized pKi by default); cycle lists follow the closed node sequence. Change the count with `--count`.

Molecule IDs accompany activations, PCA coordinates, and graphs; mismatched row orders are rejected. Pass matching input paths when changing defaults. Steps 06–09 require `--overwrite` to replace existing outputs.

Shared dataset files and activations live in `data/processed/`; layer outputs live in `data/processed/layer18/`; plots go to `figures/`. Generated files are excluded from Git. Holonomy is exploratory and does not establish Class C.
