"""Validate named or caller-supplied sets of the 15 independent rules."""
import json
from collections.abc import Mapping
from pathlib import Path

RULES = frozenset({
    'screens', 'early_dedup', 'shape', 'penalty', 'strict_floor', 'far_veto',
    'bridge', 'boundary_dedup', 'late_overlap', 'reflection', 'handoff',
    'semantic_zones', 'static', 'mirror_votes', 'short_group',
})
PROFILE_NAMES = ('core', 'greedy-add', 'greedy-remove', 'full', 'balanced')
PROFILE_DIR = Path(__file__).resolve().parents[2] / 'config' / 'profiles'


def load_profile(profile):
    if isinstance(profile, Mapping):
        definition = dict(profile)
    elif isinstance(profile, str) and profile in PROFILE_NAMES:
        definition = json.loads((PROFILE_DIR / f'{profile}.json').read_text())
    elif isinstance(profile, (str, Path)) and Path(profile).is_file():
        definition = json.loads(Path(profile).read_text())
    else:
        raise ValueError(f'Profile must be one of {", ".join(PROFILE_NAMES)}, a JSON file, or a profile mapping')
    names = definition['enabled_rules']
    enabled = frozenset(names)
    if enabled - RULES or len(enabled) != len(names):
        raise ValueError(f"Unknown or duplicate profile rules: {names}")
    return definition['name'], enabled
