# forecaster/__init__.py
"""
Forecaster-Paket für GVB-Dashboard
"""

# Explizite Exports für einfacheren Import
try:
    from .forecaster_pipeline import (
        Config as PipelineConfig,
        ModelArtifact,
        get_model_filepath,
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