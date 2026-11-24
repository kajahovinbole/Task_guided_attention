
from pathlib import Path
import re

# Change this if the images are in another location.
ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "data/raw/18okt/"

# First run: True = only show what would be deleted.
DRY_RUN = True

# Matcher for eksempel:
# IMG_3611 (1).jpeg
# IMG_3611 (2).jpeg
# IMG_3611 (3).jpeg
duplicate_pattern = re.compile(r"^(.*) \(\d+\)$")

if not IMAGE_DIR.exists():
    raise FileNotFoundError(f"Could not find folder: {IMAGE_DIR}")

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

        # Look for the original with the same name and file type.
        original = path.with_name(original_stem + path.suffix)

        if original.exists():
            print(f"{'[DRY RUN] Would delete' if DRY_RUN else 'Deleting'}: {path.name}")
            print(f"           Keeping: {original.name}")

            if not DRY_RUN:
                path.unlink()

            deleted += 1
        else:
            print(f"Keeping {path.name}: could not find original {original.name}")
            kept += 1
    else:
        kept += 1

print("\nSummary")
print(f"Copies that would be deleted / were deleted: {deleted}")
print(f"Other files kept: {kept}")

if DRY_RUN:
    print("\nNo files were deleted. Set DRY_RUN = False after reviewing the list.")
