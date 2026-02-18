import os
import glob

data_dir = "data/raw"
expected_seasons = ["2021", "2122", "2223", "2324", "2425"]

files = glob.glob(os.path.join(data_dir, "*.csv"))

coverage = {}
for f in files:
    stem = os.path.splitext(os.path.basename(f))[0]
    parts = stem.split("_")
    if len(parts) != 2:
        continue
    league, tag = parts
    coverage.setdefault(league, set()).add(tag)

print(f"{'league':<10} {'seasons found':<40} {'missing'}")
print("-" * 70)
for league in sorted(coverage):
    found   = coverage[league]
    missing = [s for s in expected_seasons if s not in found]
    found_str   = ", ".join(sorted(found))
    missing_str = ", ".join(missing) if missing else "none"
    print(f"{league:<10} {found_str:<40} {missing_str}")

all_leagues = sorted(coverage.keys())
print(f"\n{len(all_leagues)} league(s) found: {', '.join(all_leagues)}")