"""Reusable local multi-camera person tracking."""
from .profiles import PROFILE_NAMES, RULES, load_profile
from .run import run_pipeline

__all__ = ('PROFILE_NAMES', 'RULES', 'load_profile', 'run_pipeline')
