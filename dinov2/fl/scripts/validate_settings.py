#!/usr/bin/env python
"""
Validate that generated JSON files match the city_partitions.json format.
"""

import json
import sys


def validate_partition_json(json_path: str) -> bool:
    """Validate partition JSON file format.
    
    Checks:
    1. Valid JSON structure
    2. Each client has required fields: client_name, num_samples, data
    3. Data entries are [image_path, label_path] pairs
    4. Paths are relative (not absolute)
    5. Paths follow Cityscapes structure
    
    Returns:
        True if valid, False otherwise
    """
    print(f"\nValidating: {json_path}")
    print("=" * 70)
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ ERROR: Failed to load JSON: {e}")
        return False
    
    # Check overall structure
    if not isinstance(data, dict):
        print("❌ ERROR: Root should be a dictionary")
        return False
    
    n_clients = len(data)
    total_samples = 0
    errors = []
    
    for client_id_str, client_data in data.items():
        # Check client ID is numeric string
        try:
            client_id = int(client_id_str)
        except ValueError:
            errors.append(f"Client ID '{client_id_str}' is not numeric")
            continue
        
        # Check required fields
        required_fields = ['client_name', 'num_samples', 'data']
        for field in required_fields:
            if field not in client_data:
                errors.append(f"Client {client_id}: Missing field '{field}'")
                continue
        
        # Check data structure
        if not isinstance(client_data['data'], list):
            errors.append(f"Client {client_id}: 'data' should be a list")
            continue
        
        # Check num_samples matches actual data length
        actual_samples = len(client_data['data'])
        declared_samples = client_data['num_samples']
        if actual_samples != declared_samples:
            errors.append(f"Client {client_id}: num_samples={declared_samples} but data has {actual_samples} entries")
        
        total_samples += actual_samples
        
        # Check first few data entries
        for i, entry in enumerate(client_data['data'][:3]):
            if not isinstance(entry, list) or len(entry) != 2:
                errors.append(f"Client {client_id}, Sample {i}: Should be [image_path, label_path]")
                continue
            
            image_path, label_path = entry
            
            # Check paths are relative (not absolute)
            if image_path.startswith('/'):
                errors.append(f"Client {client_id}, Sample {i}: Image path should be relative, got '{image_path}'")
            if label_path.startswith('/'):
                errors.append(f"Client {client_id}, Sample {i}: Label path should be relative, got '{label_path}'")
            
            # Check Cityscapes structure
            if 'leftImg8bit' not in image_path:
                errors.append(f"Client {client_id}, Sample {i}: Image path should contain 'leftImg8bit'")
            if 'gtFine' not in label_path:
                errors.append(f"Client {client_id}, Sample {i}: Label path should contain 'gtFine'")
            if '_leftImg8bit.png' not in image_path:
                errors.append(f"Client {client_id}, Sample {i}: Image should end with '_leftImg8bit.png'")
            if '_gtFine_labelTrainIds.png' not in label_path:
                errors.append(f"Client {client_id}, Sample {i}: Label should end with '_gtFine_labelTrainIds.png'")
    
    # Print results
    if errors:
        print(f"❌ VALIDATION FAILED with {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  • {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        return False
    else:
        print("✅ VALIDATION PASSED")
        print(f"  Clients: {n_clients}")
        print(f"  Total samples: {total_samples}")
        print(f"  Format: Matches city_partitions.json")
        
        # Show sample entry
        first_client = data['0']
        print(f"\n  Sample entry (Client 0):")
        print(f"    client_name: {first_client['client_name']}")
        print(f"    num_samples: {first_client['num_samples']}")
        print(f"    data[0][0]: {first_client['data'][0][0]}")
        print(f"    data[0][1]: {first_client['data'][0][1]}")
        
        return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate partition JSON format")
    parser.add_argument("json_files", nargs='+', help="JSON files to validate")
    
    args = parser.parse_args()
    
    all_valid = True
    for json_file in args.json_files:
        valid = validate_partition_json(json_file)
        all_valid = all_valid and valid
    
    print("\n" + "=" * 70)
    if all_valid:
        print("✅ ALL FILES VALID")
        sys.exit(0)
    else:
        print("❌ SOME FILES INVALID")
        sys.exit(1)


if __name__ == "__main__":
    main()
