
from pathlib import Path
import re

# Endre denne hvis bildene ligger et annet sted.
ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "data/raw/18okt/"

# Første kjøring: True = bare vis hva som ville blitt slettet.
DRY_RUN = True

# Matcher for eksempel:
# IMG_3611 (1).jpeg
# IMG_3611 (2).jpeg
# IMG_3611 (3).jpeg
duplicate_pattern = re.compile(r"^(.*) \(\d+\)$")

if not IMAGE_DIR.exists():
    raise FileNotFoundError(f"Fant ikke mappen: {IMAGE_DIR}")

deleted = 0
kept = 0

for path in sorted(IMAGE_DIR.iterdir()):
    if not path.is_file():
        continue

    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue

    match = duplicate_pattern.match(path.stem)

    if match:
        original_stem = match.group(1)

        # Se etter originalen med samme navn og filtype.
        original = path.with_name(original_stem + path.suffix)

        if original.exists():
            print(f"{'[DRY RUN] Ville slettet' if DRY_RUN else 'Sletter'}: {path.name}")
            print(f"           Beholder: {original.name}")

            if not DRY_RUN:
                path.unlink()

            deleted += 1
        else:
            print(f"Beholder {path.name}: fant ikke originalen {original.name}")
            kept += 1
    else:
        kept += 1

print("\nOppsummering")
print(f"Kopier som ville blitt slettet / ble slettet: {deleted}")
print(f"Andre filer beholdt: {kept}")

if DRY_RUN:
    print("\nIngen filer ble slettet. Sett DRY_RUN = False når du har kontrollert listen.")
