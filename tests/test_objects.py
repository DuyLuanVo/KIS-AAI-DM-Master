#!/usr/bin/env python3
"""
Test script to check object detection loading
"""

import json
from pathlib import Path

# ===== CONFIGURATION =====
DATA_ROOT = str(Path(__file__).resolve().parent.parent / "data")
OBJ_DIR = "objects-aic25-b1/objects"


def test_object_loading():
    """Test object detection file loading"""
    print("Testing Object Detection Loading...")

    # Test with a known video
    test_video_id = "video"  # Replace with actual video ID
    test_keyframe_nums = [1, 37, 50]  # Test different keyframes

    base_path = Path(DATA_ROOT) / OBJ_DIR / test_video_id
    print(f"Looking in: {base_path}")

    if not base_path.exists():
        print(f"[ERROR] Directory not found: {base_path}")
        return False

    # List all JSON files to see naming pattern
    json_files = list(base_path.glob("*.json"))
    print(f"Found {len(json_files)} JSON files")

    # Show first few files to understand naming pattern
    for i, file in enumerate(json_files[:5]):
        print(f"   {i+1}. {file.name}")

    # Test different naming conventions
    success_count = 0
    for keyframe_num in test_keyframe_nums:
        print(f"\nTesting keyframe {keyframe_num}:")

        # Try 4-digit format: 0037.json
        # Try 3-digit format: 037.json
        path_4digit = base_path / f"{keyframe_num:04d}.json"
        path_3digit = base_path / f"{keyframe_num:03d}.json"

        found_path = None
        if path_4digit.exists():
            found_path = path_4digit
            print(f"   Found (4-digit): {path_4digit.name}")
        elif path_3digit.exists():
            found_path = path_3digit
            print(f"   Found (3-digit): {path_3digit.name}")
        else:
            print(f"   Not found: {keyframe_num:04d}.json or {keyframe_num:03d}.json")
            continue

        # Test loading and parsing
        try:
            with open(found_path, 'r') as f:
                data = json.load(f)

            scores = data.get('detection_scores', [])
            entities = data.get('detection_class_entities', [])

            print(f"   Objects: {len(scores)} detections")
            print(f"   High confidence (>0.7): {sum(1 for s in scores if float(s) > 0.7)}")

            # Show top entities
            if entities:
                high_conf_entities = [entities[i] for i, s in enumerate(scores) if float(s) > 0.5]
                unique_entities = list(set(high_conf_entities))[:5]
                print(f"   Top entities: {', '.join(unique_entities)}")

            success_count += 1

        except Exception as e:
            print(f"   Error loading {found_path}: {e}")

    print(f"\nSuccess rate: {success_count}/{len(test_keyframe_nums)}")
    return success_count > 0


def check_file_naming_pattern():
    """Check what naming pattern is used"""
    print("\nAnalyzing file naming patterns...")

    base_path = Path(DATA_ROOT) / OBJ_DIR

    if not base_path.exists():
        print(f"[ERROR] Objects directory not found: {base_path}")
        return

    # Get first video directory
    video_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    if not video_dirs:
        print("[ERROR] No video directories found")
        return

    test_dir = video_dirs[0]
    print(f"Checking directory: {test_dir.name}")

    json_files = list(test_dir.glob("*.json"))[:10]  # First 10 files

    patterns = {
        "3_digit": 0,  # 001.json, 037.json
        "4_digit": 0   # 0001.json, 0037.json
    }

    for file in json_files:
        name = file.stem  # Filename without extension
        if len(name) == 3 and name.isdigit():
            patterns["3_digit"] += 1
        elif len(name) == 4 and name.isdigit():
            patterns["4_digit"] += 1

    print(f"   3-digit format (001.json): {patterns['3_digit']} files")
    print(f"   4-digit format (0001.json): {patterns['4_digit']} files")

    # Show some examples
    for file in json_files[:5]:
        print(f"   - {file.name}")


if __name__ == "__main__":
    print("Object Detection File Test")
    print("=" * 40)

    check_file_naming_pattern()
    success = test_object_loading()

    if success:
        print("\n[SUCCESS] Object loading test passed!")
    else:
        print("\n[FAIL] Object loading test failed!")

    print("\nIf objects are not loading:")
    print("1. Check the DATA_ROOT path")
    print("2. Verify object directory structure")
    print("3. Update naming convention in loader scripts")
