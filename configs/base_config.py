import os

EPOCH_OUTPUT_BASE = os.path.join(os.environ.get('LUSTRE_HOME'), 'EpochOutput')
PROJECT_BASE = os.path.join(os.environ.get('HOME'), 'EpochSims')
RAW_DATA_BASE = os.path.join(PROJECT_BASE, 'data', 'raw')
PROCESSED_DATA_BASE = os.path.join(PROJECT_BASE, 'data', 'processed')
MEDIA_BASE = os.path.join(PROJECT_BASE, 'data', 'media')
