# Music Recommendation System

A content-based music recommender built on audio features (energy, valence,
tempo, danceability, …) of the **Top 100 streamed songs** (`Top_100_streamed_songs.csv`).

- `Music recommendation_system.ipynb` – exploratory analysis and the original Euclidean-distance approach.
- `recommender.py` – a reusable recommender module and command-line tool (needs only pandas + numpy).

## Features

| Feature | How |
|---|---|
| Similar songs to a seed song | `--song "Good 4 U"` |
| Playlist recommendations (several seeds) | repeat `--song` |
| Mood presets: happy, sad, chill, workout, party, focus | `--mood workout` |
| Target audio features | `--target energy=0.3 --target valence=0.2` |
| Euclidean or cosine similarity | `--metric cosine` |
| Feature weighting | `--weight tempo=2 --weight key=0` |
| Range filters on raw values | `--filter tempo=100:140` (open ends allowed: `tempo=120:`) |
| Search / list songs | `--search levitating`, `--list` |
| Duplicate songs collapsed | on by default; `--keep-duplicates` to disable |
| Explanations | each result lists the features most similar to the query |

Features are min-max scaled before computing distances, so no single column
(such as tempo in BPM) dominates.

## Usage

```bash
pip install -r requirements.txt

python recommender.py --song "Good 4 U" -n 5
python recommender.py --song "Good 4 U" --song "Stay The Kid LAROI" --metric cosine
python recommender.py --mood chill --filter tempo=:110
python recommender.py --target energy=0.9 --target tempo=150 --weight tempo=2
```

From Python:

```python
from recommender import MusicRecommender

rec = MusicRecommender.from_csv()
rec.recommend("Good 4 U", n=5)
rec.recommend_by_mood("happy", n=5, filters={"tempo": (100, 140)})
rec.recommend_by_features({"energy": 0.3, "valence": 0.2})
```

## Tests

```bash
python -m unittest -v
```
