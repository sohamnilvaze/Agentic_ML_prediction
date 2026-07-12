"""
Configuration file for the Agentic MIMIC Predictive Model Generation Framework.

All global constants, file paths and experiment parameters
should be defined here.

Author: Soham Vaze
"""

from pathlib import Path

# =============================================================================
# Project Directories
# =============================================================================

PROJECT_ROOT = Path("/content/drive/MyDrive/Agentic_MIMIC_POC")

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REGISTRY_DIR = DATA_DIR / "registry"

REPORT_DIR = PROJECT_ROOT / "reports"

# =============================================================================
# Raw Dataset Files
# =============================================================================

PATIENTS_FILE = RAW_DATA_DIR / "PATIENTS.csv"
ADMISSIONS_FILE = RAW_DATA_DIR / "ADMISSIONS.csv"

# =============================================================================
# Processed Dataset Files
# =============================================================================

MASTER_DATASET_VERSION = "v1"

MASTER_DATASET_FILE = (
    PROCESSED_DATA_DIR /
    f"master_dataset_{MASTER_DATASET_VERSION}.csv"
)

DATASET_SUMMARY_FILE = (
    PROCESSED_DATA_DIR /
    f"dataset_summary_{MASTER_DATASET_VERSION}.json"
)

MISSING_VALUE_REPORT_FILE = (
    PROCESSED_DATA_DIR /
    f"missing_value_report_{MASTER_DATASET_VERSION}.csv"
)

DIAGNOSIS_DISTRIBUTION_FILE = (
    PROCESSED_DATA_DIR /
    f"diagnosis_distribution_{MASTER_DATASET_VERSION}.csv"
)

# =============================================================================
# Dataset Configuration
# =============================================================================

SAMPLE_SIZE = 20_000

RANDOM_STATE = 42

# =============================================================================
# Validation Constraints
# =============================================================================

MIN_AGE = 0
MAX_AGE = 120

MIN_LENGTH_OF_STAY = 0

# =============================================================================
# Agent Configuration
# =============================================================================

SUCCESS_THRESHOLD = 0.75

MAX_DATASET_ITERATIONS = 3
MAX_MODEL_ITERATIONS = 3
MAX_TOTAL_ITERATIONS = 5

# =============================================================================
# Logging
# =============================================================================

LOG_LEVEL = "INFO"

# =============================================================================
# Future Expansion
# =============================================================================

ENABLE_NOTEEVENTS = False
ENABLE_LABEVENTS = False

MASTER_COLUMNS = [

    "subject_id",

    "hadm_id",

    "admittime",

    "dischtime",

    "admission_type",

    "admission_location",

    "discharge_location",

    "insurance",

    "language",

    "religion",

    "marital_status",

    "ethnicity",

    "diagnosis",

    "hospital_expire_flag",

    "gender",

    "dob"

]

DATASET_PROFILE_FILE = (
    PROCESSED_DATA_DIR /
    f"dataset_profile_{MASTER_DATASET_VERSION}.json"
)