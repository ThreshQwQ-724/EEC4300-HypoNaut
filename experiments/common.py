"""Shared paths and provenance; original datasets and upstream code are read-only."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / 'HypoSpace_datasets'
COMMIT = 'c69e9318577b34b5b896996571aefd4ba6053f58'
UPSTREAM = ROOT / 'upstream' / ('_HypoSpace-' + COMMIT)
RESULTS = HERE / 'results'
RESULTS.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE / '.deps'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def save(name, obj):
    path = RESULTS / name
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_upstream(domain, filename):
    # Upstream uses a shared top-level "modules" package: one domain per process.
    sys.path.insert(0, str(UPSTREAM / domain))
    spec = importlib.util.spec_from_file_location('official_benchmark', UPSTREAM / domain / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Replay:
    """Offline evaluator fixture. Never represents a trained model or makes network calls."""
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0
        self.prompts = []

    def query_with_usage(self, prompt):
        self.prompts.append(prompt)
        self.calls += 1
        return {'response': next(self.responses), 'usage': {
            'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}, 'cost': 0.0}

    def get_name(self):
        return 'OFFLINE_REPLAY_NOT_LLM'
