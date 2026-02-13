import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

evalsets_path = os.path.join(ROOT, 'evalsets')
datasets_path = os.path.join(ROOT, 'datasets')
results_path = os.path.join(ROOT, 'results')
