import h5py

def explore_h5(file_path):
    print(f"--- Exploring: {file_path} ---")
    
    try:
        with h5py.File(file_path, 'r') as f:
            
            # 1. Check for file-level metadata (attributes)
            if f.attrs.keys():
                print("Global Attributes:")
                for key, val in f.attrs.items():
                    print(f"  {key}: {val}")
            print("-" * 40)
            
            # 2. Define a visitor function to map the hierarchy
            def print_structure(name, obj):
                # Calculate indentation based on depth in the hierarchy
                indent = "  " * (name.count('/') + 1)
                
                if isinstance(obj, h5py.Group):
                    print(f"{indent}📂 Group: {name}")
                    # Print any attributes attached to this group
                    for key, val in obj.attrs.items():
                        print(f"{indent}  ↳ Attr [{key}]: {val}")
                        
                elif isinstance(obj, h5py.Dataset):
                    print(f"{indent}📊 Dataset: {name} | Shape: {obj.shape} | Type: {obj.dtype}")
                    # Print any attributes attached to this dataset
                    for key, val in obj.attrs.items():
                        print(f"{indent}  ↳ Attr [{key}]: {val}")

            # 3. Traverse the entire file tree
            f.visititems(print_structure)

    except FileNotFoundError:
        print(f"Error: Could not find the file at {file_path}")
    except OSError as e:
        print(f"Error opening the file: {e}. Is it a valid HDF5 file?")

if __name__ == "__main__":
    # Replace this with the path to your file
    target_file = "/Users/mbronars/workspace/CMU/Sp26/Planning/class_project/packages/Planning_wrapper/demos/PushBoundary/push_demos.h5" 
    explore_h5(target_file)