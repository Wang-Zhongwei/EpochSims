import os

ACCOUNT_NUMBER = "PAS0035"

LUSTRE_HOME = os.environ.get("LUSTRE_HOME")
assert LUSTRE_HOME is not None, "LUSTRE_HOME environment variable not set"

EPOCH_BASE= os.path.join(os.environ.get("HOME"), "epoch", "epoch3d")
EPOCH_OUTPUT_BASE = os.path.join(LUSTRE_HOME, "EpochOutput")

PROJECT_BASE = os.path.join(os.environ.get("HOME"), "EpochSims")
DATA_BASE = os.path.join(PROJECT_BASE, "data")
INPUT_BASE = os.path.join(PROJECT_BASE, "input")
SCRIPT_BASE = os.path.join(PROJECT_BASE, "scripts")
LOG_BASE = os.path.join(PROJECT_BASE, "logs")

CONDA_ENV_NAME = "general"