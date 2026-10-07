"""Execute each walkthrough in a fresh kernel; optionally verify expensive paths."""
from pathlib import Path
import argparse
import os
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
runtime = ROOT / 'data/notebook_runtime'
runtime.mkdir(parents=True, exist_ok=True)
os.environ['JUPYTER_RUNTIME_DIR'] = str(runtime)
os.environ['IPYTHONDIR'] = str(runtime / 'ipython')
os.environ['MPLCONFIGDIR'] = str(runtime / 'matplotlib')

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--verify-expensive', action='store_true')
args = parser.parse_args()
for path in sorted((ROOT / 'notebooks/03_modeling').glob('*.ipynb')):
    print('Executing ' + path.name, flush=True)
    notebook = nbformat.read(path, as_version=4)
    if args.verify_expensive:
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                for flag in ['REBUILD_FROM_MARTS', 'RETRAIN', 'REFIT_SELECTED']:
                    cell.source = cell.source.replace(flag + ' = False', flag + ' = True')
    NotebookClient(notebook, timeout=1800, kernel_name='python3',
                   resources={'metadata': {'path': str(path.parent)}}).execute()
    if not args.verify_expensive:
        nbformat.write(notebook, path)
    print('Passed ' + path.name, flush=True)
