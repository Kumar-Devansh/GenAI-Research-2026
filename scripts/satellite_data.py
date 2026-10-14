"""Small, bounded Sentinel-2 catalog search and labeled mask validation.

Catalog search is metadata-only. Download and segmentation require appropriately
licensed assets and genuinely labeled masks; this module does not fabricate either.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path

STAC_SEARCH = "https://stac.dataspace.copernicus.eu/v1/search"


def search_sentinel2(bbox: tuple[float, float, float, float], start: str, end: str,
                     max_cloud: float = 20, limit: int = 5, timeout: tuple[int, int] = (5, 30)) -> dict:
    if len(bbox) != 4 or not (-180 <= bbox[0] < bbox[2] <= 180 and -90 <= bbox[1] < bbox[3] <= 90):
        raise ValueError("bbox must be [west, south, east, north] in WGS84 degrees.")
    if not 0 <= max_cloud <= 100 or not 1 <= limit <= 20:
        raise ValueError("max_cloud must be 0–100 and limit must be 1–20.")
    # Validate date syntax and ordering before contacting the remote catalog.
    begin, finish = datetime.fromisoformat(start), datetime.fromisoformat(end)
    if begin > finish:
        raise ValueError("start must precede end.")
    import requests
    try:
        response = requests.post(STAC_SEARCH, json={"collections": ["sentinel-2-l2a"],
            "bbox": list(bbox), "datetime": f"{start}/{end}", "limit": limit,
            "query": {"eo:cloud_cover": {"lte": max_cloud}}}, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(f"Copernicus STAC search failed: {exc}") from exc
    features = data.get("features") if isinstance(data, dict) else None
    if not isinstance(features, list):
        raise RuntimeError("Copernicus STAC response did not contain a feature list.")
    records = [{"id": f.get("id"), "datetime": f.get("properties", {}).get("datetime"),
                "cloud_cover": f.get("properties", {}).get("eo:cloud_cover"),
                "bbox": f.get("bbox"), "assets": sorted((f.get("assets") or {}).keys())}
               for f in features]
    return {"provider": "Copernicus Data Space Ecosystem", "collection": "Sentinel-2 L2A",
            "catalog_url": STAC_SEARCH, "queried_at_utc": datetime.now(timezone.utc).isoformat(),
            "items": records, "downloaded": False}


def validate_paired_tiles(image_dir: Path, mask_dir: Path, extensions=(".tif", ".tiff", ".png")) -> list[tuple[Path, Path]]:
    """Find one-to-one pairs with equal stems; fail on missing labels."""
    images = {p.stem: p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions} if image_dir.is_dir() else {}
    masks = {p.stem: p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions} if mask_dir.is_dir() else {}
    if not images or not masks:
        raise FileNotFoundError(f"Expected labeled satellite tiles in {image_dir} and masks in {mask_dir}.")
    if images.keys() != masks.keys():
        raise ValueError(f"Image/mask names must match exactly. Missing masks: {sorted(images.keys()-masks.keys())}; orphan masks: {sorted(masks.keys()-images.keys())}.")
    return [(images[k], masks[k]) for k in sorted(images)]


def build_unet(input_shape=(256, 256, 4), classes=1):
    """Create a compact U-Net; it requires TensorFlow and provides no pretrained weights."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("Install the project ML requirements to build the U-Net.") from exc
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    skips = []
    for filters in (32, 64, 128, 256):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        skips.append(x)
        x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu")(x)
    for filters, skip in zip((256, 128, 64, 32), reversed(skips)):
        x = tf.keras.layers.UpSampling2D()(x)
        x = tf.keras.layers.Concatenate()([x, skip])
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    logits = tf.keras.layers.Conv2D(classes, 1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, logits, name="satellite_unet")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[dice_score, iou_score])
    return model


def train_unet(train_pairs, validation_pairs, test_pairs, loader, output_path: Path,
               input_shape=(256, 256, 4), epochs=30, batch_size=8):
    """Train/evaluate using caller-loaded, genuinely labeled, disjoint tile sets.

    `loader(image_path, mask_path)` must return float image HxWxC reflectance
    scaled to [0,1] and binary mask HxWx1. Explicit separate pair lists make
    split provenance the caller's responsibility and avoid random neighboring
    patch leakage hidden inside this helper.
    """
    import numpy as np
    import tensorflow as tf
    groups = [list(train_pairs), list(validation_pairs), list(test_pairs)]
    if any(not group for group in groups):
        raise ValueError("Provide non-empty train, validation, and test image/mask sets.")
    stems = [{Path(image).stem for image, _ in group} for group in groups]
    if (stems[0] & stems[1]) or (stems[0] & stems[2]) or (stems[1] & stems[2]):
        raise ValueError("Train, validation, and test tile identifiers must be disjoint.")

    def load_group(pairs):
        images, masks = [], []
        for image_path, mask_path in pairs:
            image, mask = loader(image_path, mask_path)
            image, mask = np.asarray(image, np.float32), np.asarray(mask, np.float32)
            if image.ndim != 3 or image.shape[-1] != input_shape[-1]:
                raise ValueError(f"Image {image_path} must have HxWx{input_shape[-1]} channels.")
            if mask.ndim == 2:
                mask = mask[..., None]
            if mask.ndim != 3 or mask.shape[-1] != 1:
                raise ValueError(f"Mask {mask_path} must be a single-channel binary mask.")
            if image.size == 0 or mask.size == 0 or not np.isfinite(image).all() or not np.isfinite(mask).all():
                raise ValueError(f"Non-finite or empty tile pair: {image_path}")
            if image.min() < 0 or image.max() > 1 or mask.min() < 0 or mask.max() > 1:
                raise ValueError("Scale reflectance and binary masks to [0, 1] in the loader.")
            images.append(tf.image.resize(image, input_shape[:2]).numpy())
            masks.append(tf.image.resize(mask, input_shape[:2], method="nearest").numpy())
        return np.stack(images), np.stack(masks)

    x_train, y_train = load_group(groups[0])
    x_val, y_val = load_group(groups[1])
    x_test, y_test = load_group(groups[2])
    model = build_unet(input_shape)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)]
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs,
                        batch_size=batch_size, callbacks=callbacks, verbose=0)
    results = model.evaluate(x_test, y_test, return_dict=True, verbose=0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    return {"test_metrics": {k: float(v) for k, v in results.items()},
            "epochs_trained": len(history.history.get("loss", [])),
            "n_train": len(x_train), "n_validation": len(x_val), "n_test": len(x_test),
            "model_path": str(output_path)}


def dice_score(y_true, y_pred):
    import tensorflow as tf
    yt, yp = tf.reshape(tf.cast(y_true, tf.float32), [-1]), tf.reshape(tf.cast(y_pred >= .5, tf.float32), [-1])
    return (2 * tf.reduce_sum(yt * yp) + 1e-6) / (tf.reduce_sum(yt) + tf.reduce_sum(yp) + 1e-6)


def iou_score(y_true, y_pred):
    import tensorflow as tf
    yt, yp = tf.reshape(tf.cast(y_true, tf.float32), [-1]), tf.reshape(tf.cast(y_pred >= .5, tf.float32), [-1])
    return (tf.reduce_sum(yt * yp) + 1e-6) / (tf.reduce_sum(yt) + tf.reduce_sum(yp) - tf.reduce_sum(yt * yp) + 1e-6)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Search Sentinel-2 metadata; does not download imagery.")
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--start", required=True, help="ISO date or datetime")
    parser.add_argument("--end", required=True, help="ISO date or datetime")
    parser.add_argument("--max-cloud", type=float, default=20)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(search_sentinel2(tuple(args.bbox), args.start, args.end,
                                      max_cloud=args.max_cloud, limit=args.limit), indent=2))

