"""Part 02 - Step 2: YOLOv8 classification on your sky_images (clear_sky / partly_cloudy / heavy_clouds)."""
import random, shutil
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
SRC, DST = ROOT / "data" / "sky_images", ROOT / "data" / "sky_split"

if DST.exists():
    shutil.rmtree(DST)
random.seed(42)
for cls in [d for d in SRC.iterdir() if d.is_dir()]:
    files = [f for f in cls.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    random.shuffle(files)
    cut = int(len(files) * 0.8)
    for name, part in [("train", files[:cut]), ("val", files[cut:])]:
        (DST / name / cls.name).mkdir(parents=True, exist_ok=True)
        for f in part:
            shutil.copy(f, DST / name / cls.name / f.name)

model = YOLO("yolov8n-cls.pt")   # downloads automatically on first run
model.train(data=str(DST), epochs=15, imgsz=128, project=str(ROOT / "outputs"), name="yolo_sky", exist_ok=True)
print(model.val())
model.export(format="onnx")
