#!/usr/bin/env python3
"""Fix type hints in ptlflow for Python 3.9 compatibility"""
import re
import sys
from pathlib import Path

def fix_union_type_hints(file_path):
    """Replace new-style union type hints with typing.Union for Python 3.9 compatibility"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern to match: tuple[type, type] | None or similar
    # This regex looks for patterns like: type[...] | None
    pattern = r'(\w+\[[^\]]+\])\s*\|\s*None'
    replacement = r'Union[\1, None]'
    content = re.sub(pattern, replacement, content)
    
    # Also handle simple type | None patterns
    simple_pattern = r':\s*(\w+)\s*\|\s*None'
    simple_replacement = r': Union[\1, None]'
    content = re.sub(simple_pattern, simple_replacement, content)
    
    # If we made changes and Union is used, add the import
    if content != original_content and 'Union[' in content:
        # Check if typing import exists
        if 'from typing import' in content:
            # Add Union to existing import if not already there
            if 'Union' not in content:
                content = re.sub(
                    r'from typing import ([^)]+)',
                    r'from typing import \1, Union',
                    content,
                    count=1
                )
        else:
            # Add new typing import at the top
            lines = content.split('\n')
            import_added = False
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    lines.insert(i, 'from typing import Union')
                    import_added = True
                    break
            if not import_added:
                lines.insert(0, 'from typing import Union')
            content = '\n'.join(lines)
    
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    # Find the ptlflow installation path
    ptlflow_path = Path("/research/cbim/vast/sf895/miniforge3/envs/geometry/lib/python3.9/site-packages/ptlflow")
    
    if not ptlflow_path.exists():
        print(f"Error: ptlflow not found at {ptlflow_path}")
        sys.exit(1)
    
    # Fix the specific file that's causing issues
    matchflow_file = ptlflow_path / "models" / "matchflow" / "matchflow.py"
    
    if matchflow_file.exists():
        print(f"Fixing type hints in {matchflow_file}")
        if fix_union_type_hints(matchflow_file):
            print("✓ Fixed matchflow.py")
        else:
            print("No changes needed in matchflow.py")
    
    # Also check other files that might have similar issues
    for py_file in ptlflow_path.rglob("*.py"):
        try:
            if fix_union_type_hints(py_file):
                print(f"✓ Fixed {py_file.relative_to(ptlflow_path)}")
        except Exception as e:
            print(f"Error processing {py_file}: {e}")
    
    print("\nType hint fixes completed!")

if __name__ == "__main__":
    main()