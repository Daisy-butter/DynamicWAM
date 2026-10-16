#!/usr/bin/env python3
"""Select 256 real physical-grid samples for bounded GPU smoke, CPU only."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dynamicwam.training.data.smoke_grid import prepare_smoke_subset

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to(Path('/SSD_DISK/users/wuruihan').resolve()):
        raise ValueError('smoke data belongs on verified SSD')
    print(json.dumps(prepare_smoke_subset(args.parent_root, args.output_dir), indent=2))

if __name__ == '__main__':
    main()
