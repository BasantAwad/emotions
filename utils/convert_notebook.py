import json
import sys
import os

def convert_notebook(input_path, output_path):
    print(f"Converting {input_path} to {output_path}...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        f_out.write(f"# Converted from {os.path.basename(input_path)}\n\n")
        
        for cell in nb.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = cell.get('source', [])
                if not source:
                    continue
                
                f_out.write("# %% [code]\n")
                for line in source:
                    # Comment out magic commands
                    if line.strip().startswith('!') or line.strip().startswith('%'):
                        f_out.write(f"# {line}")
                    else:
                        f_out.write(line)
                f_out.write("\n\n")
            elif cell.get('cell_type') == 'markdown':
                source = cell.get('source', [])
                if not source:
                    continue
                f_out.write("# %% [markdown]\n")
                for line in source:
                     f_out.write(f"# {line}")
                f_out.write("\n\n")

    print(f"Conversion complete: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_notebook.py <input.ipynb> <output.py>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    convert_notebook(input_file, output_file)
