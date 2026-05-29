import sys
import os
import numpy as np

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.vision.svg_parser import SVGParser

def main():
    print("Testing SVGParser...")
    svg_content = """<?xml version="1.0" encoding="utf-8"?>
    <svg version="1.1" xmlns="http://www.w3.org/2000/svg" width="500" height="500">
        <rect x="10" y="20" width="100" height="150" transform="translate(5, 5)" />
        <circle cx="200" cy="200" r="50" transform="scale(2)" />
        <polygon points="50,50 100,50 100,100" />
        <path d="M 10 10 L 20 20 C 30 30, 40 10, 50 20 Z" transform="matrix(1 0 0 1 10 10)" />
        <path d="M 100 100 L 200 200" /> <!-- Open path -->
    </svg>
    """
    
    # Save to a temporary file
    temp_path = "scratch/temp_test.svg"
    os.makedirs("scratch", exist_ok=True)
    with open(temp_path, "w") as f:
        f.write(svg_content)
        
    try:
        parser = SVGParser(bezier_resolution=10)
        elements = parser.parse_file(temp_path)
        
        print(f"Successfully parsed {len(elements)} elements:")
        for idx, elem in enumerate(elements):
            print(f"Element #{idx+1}: type={elem['type']}, points_count={len(elem['points'])}, closed={elem['closed']}")
            
    except Exception as e:
        print(f"FAILED with error: {e}", file=sys.stderr)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    main()
