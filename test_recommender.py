"""Tests for recommender.py. Run with: python -m unittest -v"""

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout

import pandas as pd

from recommender import FEATURES, MOOD_PRESETS, MusicRecommender, main


def toy_data() -> pd.DataFrame:
    base = {f: 0.5 for f in FEATURES}
    rows = []
    for name, energy, valence, tempo in [
        ("Alpha Song", 0.10, 0.10, 80),
        ("Alpha Remix", 0.12, 0.11, 82),
        ("Beta Track", 0.90, 0.90, 150),
        ("Gamma Tune", 0.50, 0.50, 120),
    ]:
        rows.append({**base, "name": name, "energy": energy, "valence": valence, "tempo": tempo})
    return pd.DataFrame(rows)


class RecommenderTests(unittest.TestCase):
    def setUp(self):
        self.rec = MusicRecommender(toy_data())

    def test_search_case_insensitive(self):
        self.assertEqual(list(self.rec.search("alpha")["name"]), ["Alpha Song", "Alpha Remix"])

    def test_find_index_exact_beats_partial(self):
        self.assertEqual(self.rec.find_index("alpha song"), 0)

    def test_find_index_ambiguous_and_missing(self):
        with self.assertRaises(KeyError):
            self.rec.find_index("Alpha", strict=True)
        with self.assertRaises(KeyError):
            self.rec.find_index("Nope")

    def test_ambiguous_name_uses_first_match_when_lenient(self):
        with redirect_stderr(io.StringIO()):
            self.assertEqual(self.rec.find_index("Alpha"), 0)

    def test_dedupe_collapses_identical_songs(self):
        data = pd.concat([toy_data(), toy_data().iloc[[2]].assign(name="Beta Track (feat. X)")])
        rec = MusicRecommender(data)
        names = list(rec.recommend("Alpha Song", n=5)["name"])
        self.assertEqual(sum(n.startswith("Beta") for n in names), 1)
        names = list(rec.recommend("Alpha Song", n=5, dedupe=False)["name"])
        self.assertEqual(sum(n.startswith("Beta") for n in names), 2)

    def test_recommend_excludes_seed_and_ranks_nearest_first(self):
        result = self.rec.recommend("Alpha Song", n=3)
        self.assertNotIn("Alpha Song", list(result["name"]))
        self.assertEqual(result.iloc[0]["name"], "Alpha Remix")
        self.assertTrue(result["distance"].is_monotonic_increasing)

    def test_cosine_metric(self):
        result = self.rec.recommend(0, n=1, metric="cosine")
        self.assertEqual(result.iloc[0]["name"], "Alpha Remix")

    def test_playlist_seeds(self):
        result = self.rec.recommend(["Alpha Song", "Beta Track"], n=2)
        self.assertEqual(result.iloc[0]["name"], "Gamma Tune")

    def test_filters(self):
        result = self.rec.recommend("Alpha Song", n=5, filters={"tempo": (100, 200)})
        self.assertEqual(set(result["name"]), {"Beta Track", "Gamma Tune"})

    def test_recommend_by_features(self):
        result = self.rec.recommend_by_features({"energy": 0.9, "valence": 0.9}, n=1)
        self.assertEqual(result.iloc[0]["name"], "Beta Track")
        self.assertIn("energy", result.iloc[0]["similar_on"])

    def test_weights_change_ranking(self):
        # With all weight on tempo, Gamma (120) is closest to a 115 BPM target.
        self.rec.set_weights({f: 0 for f in FEATURES if f != "tempo"})
        result = self.rec.recommend_by_features({"tempo": 115, "energy": 0.1}, n=1)
        self.assertEqual(result.iloc[0]["name"], "Gamma Tune")

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            self.rec.set_weights({"bogus": 1})
        with self.assertRaises(ValueError):
            self.rec.recommend_by_mood("angry")
        with self.assertRaises(ValueError):
            self.rec.recommend(0, metric="manhattan")

    def test_all_moods_work_on_real_data(self):
        rec = MusicRecommender.from_csv()
        self.assertEqual(len(rec.data), 100)
        for mood in MOOD_PRESETS:
            self.assertEqual(len(rec.recommend_by_mood(mood, n=3)), 3)


class CliTests(unittest.TestCase):
    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_song(self):
        code, out, _ = self.run_cli("--song", "Good 4 U", "-n", "3")
        self.assertEqual(code, 0)
        self.assertEqual(len(out.strip().splitlines()), 4)  # header + 3 rows

    def test_mood_with_filter(self):
        code, out, _ = self.run_cli("--mood", "workout", "--filter", "tempo=120:")
        self.assertEqual(code, 0)
        self.assertIn("distance", out)

    def test_unknown_song_is_error(self):
        code, _, err = self.run_cli("--song", "zzzz not a song")
        self.assertEqual(code, 2)
        self.assertIn("No song matching", err)


if __name__ == "__main__":
    unittest.main()
