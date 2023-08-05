import os

ACCOUNT_NUMBER = "PAS0035"

LUSTRE_HOME = os.environ.get("LUSTRE_HOME")
assert LUSTRE_HOME is not None, "LUSTRE_HOME environment variable not set"

EPOCH_OUTPUT_BASE = os.path.join(LUSTRE_HOME, "EpochOutput")
EPOCH_BASE= os.path.join(os.environ.get("HOME"), "epoch", "epoch3d")
PROJECT_BASE = os.path.join(os.environ.get("HOME"), "EpochSims")
RAW_DATA_BASE = os.path.join(PROJECT_BASE, "data", "raw")
PROCESSED_DATA_BASE = os.path.join(PROJECT_BASE, "data", "processed")
MEDIA_BASE = os.path.join(PROJECT_BASE, "data", "media")
INPUT_BASE = os.path.join(PROJECT_BASE, "input")
SCRIPT_BASE = os.path.join(PROJECT_BASE, "scripts")
LOG_BASE = os.path.join(PROJECT_BASE, "logs")