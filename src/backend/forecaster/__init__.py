# forecaster/__init__.py
"""
Forecaster-Paket für GVB-Dashboard
"""

# Explizite Exports für einfacheren Import
from src.backend.forecaster.core.model.model_management import get_model_filepath
from src.backend.forecaster.core.model.model import ModelArtifact
from src.backend.forecaster.core.config import Config as PipelineConfig


try:
    from src.backend.forecaster.forecaster_pipeline import (
        run_production_pipeline,
    )
    
    __all__ = [
        'PipelineConfig',
        'ModelArtifact', 
        'get_model_filepath',
        'run_production_pipeline',
    ]
    
    print("[forecaster.__init__] ✓ Pipeline-Module erfolgreich geladen")
    
except Exception as e:
    print(f"[forecaster.__init__] ✗ Import-Fehler: {e}")
    raise