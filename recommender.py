"""Content-based music recommender built on the Top 100 streamed songs dataset.

Features
--------
* Search songs by (partial, case-insensitive) name.
* Recommend similar songs to one seed song, or to a whole playlist of seeds
  (the playlist "centroid" is used as the query).
* Recommend songs matching a mood preset (happy, chill, workout, sad, party,
  focus) or arbitrary target audio features (e.g. ``energy=0.9``).
* Euclidean or cosine similarity on min-max scaled audio features.
* Per-feature weights to emphasise what matters to you (e.g. ``tempo=2``).
* Duplicate entries of the same song are collapsed in results.
* Explanations: for every recommendation, the features most similar to the
  query.

Only pandas and numpy are required. Run ``python recommender.py --help`` for
the command line interface.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).with_name("Top_100_streamed_songs.csv")

FEATURES = [
    "duration",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "danceability",
]

# Mood presets expressed in raw feature units (0-1 for Spotify-style ratios,
# BPM for tempo). Only the listed features are used for matching.
MOOD_PRESETS: dict[str, dict[str, float]] = {
    "happy": {"valence": 0.85, "energy": 0.75, "danceability": 0.75, "mode": 1},
    "sad": {"valence": 0.2, "energy": 0.35, "acousticness": 0.6},
    "chill": {"energy": 0.35, "acousticness": 0.6, "tempo": 90, "loudness": -9},
    "workout": {"energy": 0.9, "tempo": 140, "danceability": 0.75, "loudness": -4},
    "party": {"danceability": 0.85, "energy": 0.8, "valence": 0.75},
    "focus": {"speechiness": 0.04, "energy": 0.45, "instrumentalness": 0.5, "liveness": 0.1},
}

METRICS = ("euclidean", "cosine")


class MusicRecommender:
    """Content-based recommender over a table of songs and audio features."""

    def __init__(
        self,
        data: pd.DataFrame,
        features: list[str] | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.features = list(features or FEATURES)
        missing = [c for c in ["name", *self.features] if c not in data.columns]
        if missing:
            raise ValueError(f"Dataset is missing columns: {missing}")

        self.data = data.reset_index(drop=True).copy()
        raw = self.data[self.features].astype(float)
        self._min = raw.min()
        span = raw.max() - self._min
        self._span = span.replace(0, 1.0)  # avoid division by zero for constant columns
        self.scaled = ((raw - self._min) / self._span).to_numpy()
        self.set_weights(weights)

    # ------------------------------------------------------------------ setup
    @classmethod
    def from_csv(cls, path: str | Path = DEFAULT_DATA_PATH, **kwargs) -> "MusicRecommender":
        return cls(pd.read_csv(path), **kwargs)

    def set_weights(self, weights: dict[str, float] | None) -> None:
        """Set per-feature weights (unlisted features keep weight 1)."""
        weights = weights or {}
        unknown = set(weights) - set(self.features)
        if unknown:
            raise ValueError(f"Unknown feature(s) in weights: {sorted(unknown)}")
        if any(w < 0 for w in weights.values()):
            raise ValueError("Weights must be non-negative")
        self.weights = np.array([float(weights.get(f, 1.0)) for f in self.features])

    # ----------------------------------------------------------------- search
    def search(self, query: str) -> pd.DataFrame:
        """Return songs whose name contains ``query`` (case-insensitive)."""
        mask = self.data["name"].str.contains(query, case=False, regex=False, na=False)
        return self.data.loc[mask, ["name"]]

    def find_index(self, song: str | int, strict: bool = False) -> int:
        """Resolve a row index or (partial) song name to a single row index.

        An exact (case-insensitive) name match wins. Otherwise a partial match
        is used; if several songs match, the first is chosen unless ``strict``
        is set, in which case a ``KeyError`` lists the candidates.
        """
        if isinstance(song, (int, np.integer)):
            if not 0 <= song < len(self.data):
                raise IndexError(f"Song index {song} out of range 0..{len(self.data) - 1}")
            return int(song)
        exact = self.data.index[self.data["name"].str.lower() == song.lower()]
        if len(exact):
            return int(exact[0])
        matches = self.search(song)
        if matches.empty:
            raise KeyError(f"No song matching {song!r}")
        if len(matches) > 1:
            names = "; ".join(matches["name"].head(5))
            if strict:
                raise KeyError(f"{song!r} is ambiguous, matches: {names}")
            print(f"note: {song!r} matched {len(matches)} songs, using {matches['name'].iloc[0]!r}",
                  file=sys.stderr)
        return int(matches.index[0])

    # -------------------------------------------------------------- distances
    def _scale_targets(self, targets: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        unknown = set(targets) - set(self.features)
        if unknown:
            raise ValueError(f"Unknown feature(s): {sorted(unknown)}")
        vec = np.zeros(len(self.features))
        mask = np.zeros(len(self.features), dtype=bool)
        for i, f in enumerate(self.features):
            if f in targets:
                vec[i] = (float(targets[f]) - self._min[f]) / self._span[f]
                mask[i] = True
        return vec, mask

    def _distances(self, query: np.ndarray, mask: np.ndarray | None, metric: str) -> np.ndarray:
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}")
        w = self.weights if mask is None else self.weights * mask
        X = self.scaled * np.sqrt(w)
        q = query * np.sqrt(w)
        if metric == "euclidean":
            return np.linalg.norm(X - q, axis=1)
        norms = np.linalg.norm(X, axis=1) * np.linalg.norm(q)
        sims = np.divide(X @ q, norms, out=np.zeros(len(X)), where=norms > 0)
        return 1.0 - sims

    def _explain(self, query: np.ndarray, mask: np.ndarray | None, rows: np.ndarray, top: int = 3) -> list[str]:
        diffs = np.abs(self.scaled[rows] - query)
        if mask is not None:
            diffs[:, ~mask] = np.inf
        out = []
        for row in diffs:
            order = [i for i in np.argsort(row) if np.isfinite(row[i])][:top]
            out.append(", ".join(self.features[i] for i in order))
        return out

    def _rank(self, query, mask, n, metric, exclude, filters, dedupe=True) -> pd.DataFrame:
        dist = self._distances(query, mask, metric)
        candidates = np.ones(len(self.data), dtype=bool)
        candidates[list(exclude)] = False
        for feature, (lo, hi) in (filters or {}).items():
            if feature not in self.data.columns:
                raise ValueError(f"Unknown filter feature: {feature}")
            col = self.data[feature].to_numpy(dtype=float)
            candidates &= (col >= lo) & (col <= hi)
        idx = np.flatnonzero(candidates)
        idx = idx[np.argsort(dist[idx], kind="stable")]
        if dedupe:
            # The dataset lists some songs several times under different names
            # (e.g. "feat." placement) with identical audio features; keep one.
            seen = {self.scaled[i].round(6).tobytes() for i in exclude}
            kept = []
            for i in idx:
                key = self.scaled[i].round(6).tobytes()
                if key not in seen:
                    seen.add(key)
                    kept.append(i)
            idx = np.array(kept, dtype=int)
        idx = idx[:n]
        result = self.data.loc[idx, ["name", *self.features]].copy()
        result["distance"] = dist[idx].round(4)
        result["similar_on"] = self._explain(query, mask, idx)
        return result

    # ---------------------------------------------------------- recommending
    def recommend(
        self,
        seeds: str | int | list[str | int],
        n: int = 5,
        metric: str = "euclidean",
        filters: dict[str, tuple[float, float]] | None = None,
        dedupe: bool = True,
    ) -> pd.DataFrame:
        """Recommend ``n`` songs similar to one seed song or a playlist of seeds.

        ``filters`` restricts candidates by raw feature ranges, e.g.
        ``{"tempo": (100, 140)}``. With ``dedupe`` (default) duplicate
        entries of the same song are collapsed.
        """
        if not isinstance(seeds, list):
            seeds = [seeds]
        if not seeds:
            raise ValueError("At least one seed song is required")
        idx = sorted({self.find_index(s) for s in seeds})
        query = self.scaled[idx].mean(axis=0)
        return self._rank(query, None, n, metric, idx, filters, dedupe)

    def recommend_by_features(
        self,
        targets: dict[str, float],
        n: int = 5,
        metric: str = "euclidean",
        filters: dict[str, tuple[float, float]] | None = None,
        dedupe: bool = True,
    ) -> pd.DataFrame:
        """Recommend songs closest to target raw feature values (only those given)."""
        if not targets:
            raise ValueError("At least one target feature is required")
        query, mask = self._scale_targets(targets)
        return self._rank(query, mask, n, metric, [], filters, dedupe)

    def recommend_by_mood(self, mood: str, n: int = 5, metric: str = "euclidean", **kwargs) -> pd.DataFrame:
        """Recommend songs matching a mood preset (see ``MOOD_PRESETS``)."""
        try:
            targets = MOOD_PRESETS[mood.lower()]
        except KeyError:
            raise ValueError(f"Unknown mood {mood!r}; choose from {sorted(MOOD_PRESETS)}") from None
        return self.recommend_by_features(targets, n=n, metric=metric, **kwargs)


# ---------------------------------------------------------------------- CLI
def _parse_pairs(items: list[str] | None, kind: str) -> dict[str, str]:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"{kind} must look like feature=value, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _parse_range(text: str) -> tuple[float, float]:
    lo, _, hi = text.partition(":")
    return (float(lo) if lo else -np.inf, float(hi) if hi else np.inf)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Content-based music recommender")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="CSV dataset path")
    parser.add_argument("-n", type=int, default=5, help="number of recommendations")
    parser.add_argument("--metric", choices=METRICS, default="euclidean")
    parser.add_argument("--weight", action="append", metavar="FEATURE=W", help="feature weight, repeatable")
    parser.add_argument("--keep-duplicates", action="store_true",
                        help="do not collapse duplicate entries of the same song")
    parser.add_argument("--filter", action="append", metavar="FEATURE=LO:HI",
                        help="restrict candidates to a raw feature range, e.g. tempo=100:140")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--song", action="append", help="seed song name (repeat for a playlist)")
    group.add_argument("--mood", choices=sorted(MOOD_PRESETS))
    group.add_argument("--target", action="append", metavar="FEATURE=VALUE",
                       help="target feature value, repeatable, e.g. energy=0.9")
    group.add_argument("--search", help="list songs whose name contains this text")
    group.add_argument("--list", action="store_true", help="list all songs")
    args = parser.parse_args(argv)

    try:
        weights = {k: float(v) for k, v in _parse_pairs(args.weight, "--weight").items()}
        filters = {k: _parse_range(v) for k, v in _parse_pairs(args.filter, "--filter").items()}
        dedupe = not args.keep_duplicates
        rec = MusicRecommender.from_csv(args.data, weights=weights)

        if args.list or args.search is not None:
            table = rec.data[["name"]] if args.list else rec.search(args.search)
            print(table.to_string() if not table.empty else "No matching songs.")
            return 0

        if args.song:
            result = rec.recommend(args.song, n=args.n, metric=args.metric, filters=filters, dedupe=dedupe)
        elif args.mood:
            result = rec.recommend_by_mood(args.mood, n=args.n, metric=args.metric, filters=filters, dedupe=dedupe)
        else:
            targets = {k: float(v) for k, v in _parse_pairs(args.target, "--target").items()}
            result = rec.recommend_by_features(targets, n=args.n, metric=args.metric, filters=filters, dedupe=dedupe)
    except (KeyError, IndexError, ValueError, argparse.ArgumentTypeError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f"error: {message}", file=sys.stderr)
        return 2

    if result.empty:
        print("No songs match the given filters.")
    else:
        print(result[["name", "distance", "similar_on"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
