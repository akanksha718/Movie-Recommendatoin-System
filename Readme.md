# Movie Recommendation System (TMDB 5000)

A simple content-based movie recommendation project built from the TMDB 5000 dataset. The current notebook cleans and merges movie and credits data, extracts relevant textual features, and constructs a consolidated "tag" for each movie. This tag can be vectorized (e.g., TF‑IDF/CountVectorizer) to compute similarities for recommendations.
<a href="https://movie-recomandation.streamlit.app/">Live Preview</a>
## Overview
- **Goal:** Recommend similar movies using content-based features from TMDB metadata.
- **Data:** TMDB 5000 Movies and Credits CSVs.
- **Approach (current):**
  - Load movies and credits, merge on `title`.
  - Parse structured fields (`genres`, `keywords`, `cast`, `crew`) from JSON-like strings.
  - Keep director, top cast, keywords, genres, and overview terms.
  - Normalize tokens (lowercase and remove spaces) and build a combined `tag` field.
- **Next steps:** Vectorize `tag` (CountVectorizer/TF‑IDF) and use cosine similarity to generate recommendations.

## Repository Structure
- `movie-recommander.ipynb` — main Jupyter Notebook with data prep and tag construction.
- `tmdb_5000_movies.csv` — movie metadata.
- `tmdb_5000_credits.csv` — cast and crew metadata.
- `LICENSE` — project license.

## Requirements
- Python 3.9+ (3.10/3.11 also fine)
- Jupyter Notebook
- Python packages: `pandas`, `numpy`
- Optional (for recommendation stage): `scikit-learn`

## Setup (Windows)
```powershell
# From the project root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install jupyter pandas numpy scikit-learn
```

## Data
Place the following CSVs in the project root (already included here):
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

## Usage
```powershell
jupyter notebook
```
Then open `movie-recommander.ipynb` and run the cells in order. The notebook will:
- Read and merge the datasets
- Parse genres/keywords/cast/crew
- Build a normalized `tag` text for each movie

```

## Notes
- If you modify paths, ensure the CSV filenames match the code in the notebook.
- For reproducibility, consider pinning package versions (e.g., via a `requirements.txt`).

## License
This project is licensed under the terms found in `LICENSE`.

## Acknowledgements
- Dataset originally provided via the TMDB 5000 Movies/Credits dataset (commonly mirrored on Kaggle).
