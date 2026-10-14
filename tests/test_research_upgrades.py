"""Fast, offline tests for weather-data boundaries and experimental workflows."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from knowledge_ingest import chunk_pages
from rainfall_models import build_rainfall_frame
from satellite_data import search_sentinel2, validate_paired_tiles
from weather_sources import WeatherSourceError, imd_current_weather, open_meteo_forecast


class ChunkTests(unittest.TestCase):
    def test_chunks_keep_page_and_overlap(self):
        parts = chunk_pages([(3, "abcdefghij")], size=6, overlap=2)
        self.assertEqual([p["page"] for p in parts], [3, 3])
        self.assertEqual(parts[0]["text"][-2:], parts[1]["text"][:2])

    def test_invalid_overlap_rejected(self):
        with self.assertRaises(ValueError):
            chunk_pages([(1, "x")], size=2, overlap=2)


class RainfallFeatureTests(unittest.TestCase):
    def test_target_is_next_day_and_predictors_are_current_or_lagged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weather.csv"
            path.write_text("date,rainfall,temp_mean,humidity,pressure,cloud_cover\n" + "".join(
                f"2024-01-{i:02d},{(i - 1) % 3},{10+i},50,1000,20\n" for i in range(1, 21)), encoding="utf-8")
            frame, predictors = build_rainfall_frame(path)
            row = frame.iloc[0]
            self.assertEqual(row.target_rainfall_next_day, 2)
            self.assertEqual(row.target_date.date().isoformat(), "2024-01-09")
            self.assertNotIn("target_rainfall_next_day", predictors)
            self.assertNotIn("target_date", predictors)


class SatelliteValidationTests(unittest.TestCase):
    def test_rejects_reversed_dates_without_network(self):
        with self.assertRaises(ValueError):
            search_sentinel2((77, 28, 78, 29), "2024-03-02", "2024-03-01")

    def test_pairing_requires_matching_labeled_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            images, masks = Path(directory) / "images", Path(directory) / "masks"
            images.mkdir(); masks.mkdir()
            (images / "tile.png").touch()
            with self.assertRaises(FileNotFoundError):
                validate_paired_tiles(images, masks)


class WeatherSourceTests(unittest.TestCase):
    @patch("weather_sources._get_json")
    def test_open_meteo_has_model_provenance(self, get_json):
        get_json.return_value = {"timezone": "Asia/Kolkata", "current": {"temperature_2m": 30}, "daily": {"time": []}}
        _open = __import__("weather_sources")._open_meteo_cached
        _open.cache_clear()
        result = open_meteo_forecast()
        self.assertEqual(result["provider"], "Open-Meteo")
        self.assertEqual(result["product"], "numerical weather model")
        _open.cache_clear()

    @patch.dict("os.environ", {}, clear=True)
    def test_imd_requires_operator_credentials(self):
        with self.assertRaises(WeatherSourceError):
            imd_current_weather()


if __name__ == "__main__":
    unittest.main()
