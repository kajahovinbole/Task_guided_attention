
from pathlib import Path
import shutil

# Mappen som inneholder alle undermappene med bilder
SOURCE_DIR = Path("/Users/kajahovinbole/Downloads/dataset/pointing")

# Mappen alle bildene skal samles i
OUTPUT_DIR = Path("data/andrea/")

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic"
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

count = 0
skipped = 0

for path in SOURCE_DIR.rglob("*"):
    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    # Behold filnavnet, men unngå å overskrive filer med samme navn.
    destination = OUTPUT_DIR / path.name

    if destination.exists():
        destination = OUTPUT_DIR / f"{path.parent.name}_{path.name}"
        version = 2

        while destination.exists():
            destination = OUTPUT_DIR / f"{path.parent.name}_{version}_{path.name}"
            version += 1

    try:
        shutil.copy2(path, destination)
        count += 1
    except OSError as e:
        print(f"Kunne ikke kopiere {path}: {e}")
        skipped += 1

print(f"Ferdig! Kopierte {count} bilder til {OUTPUT_DIR}")
if skipped:
    print(f"Kunne ikke kopiere {skipped} filer.")
