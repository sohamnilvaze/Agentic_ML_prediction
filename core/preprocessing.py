"""
Master Dataset Preprocessor

This module is responsible for creating the master dataset that serves as the
single source of truth for the Agentic MIMIC Predictive Model Generation
Framework.

Responsibilities
----------------
1. Load raw MIMIC tables.
2. Validate raw datasets.
3. Merge admissions and patients.
4. Engineer generic features.
5. Generate exploratory reports.
6. Validate processed dataset.
7. Save all artifacts.

NOTE
----
This module DOES NOT perform task-specific preprocessing such as
    - Feature selection
    - Missing value imputation
    - Encoding
    - Scaling
    - Train/Test split

Those responsibilities belong to the Dataset Builder Agent.

Author: Soham Vaze
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from core.config import (
    PATIENTS_FILE,
    ADMISSIONS_FILE,
    MASTER_DATASET_FILE,
    DATASET_SUMMARY_FILE,
    DIAGNOSIS_DISTRIBUTION_FILE,
    MISSING_VALUE_REPORT_FILE,
    PROCESSED_DATA_DIR,
    REPORT_DIR,
    RANDOM_STATE,
    SAMPLE_SIZE,
    MASTER_COLUMNS,
    DATASET_PROFILE_FILE
)

from core.utils import (
    current_timestamp,
    ensure_directory_exists,
    log_info,
    log_error,
    save_json,
)





class MasterDatasetPreprocessor:
    """
    Creates the master dataset for the complete agentic framework.

    Internal Workflow

        Load Raw Data
                ↓
        Validate Raw Data
                ↓
           Merge Tables
                ↓
          Select Columns
                ↓
       Engineer Features
                ↓
     Standardize Datatypes
                ↓
      Generate EDA Reports
                ↓
      Validate Master Dataset
                ↓
          Sample Dataset
                ↓
         Save All Artifacts
    """

    ####################################################################
    # Constructor
    ####################################################################

    def __init__(self):

        # Raw datasets

        self.patients_df: Optional[pd.DataFrame] = None

        self.admissions_df: Optional[pd.DataFrame] = None

        # Final merged dataframe

        self.master_df: Optional[pd.DataFrame] = None

        # Dataset metadata

        self.summary = {}

        log_info("MasterDatasetPreprocessor initialized.")



    ####################################################################
    # Pipeline
    ####################################################################

    def run_pipeline(self) -> None:
        """
        Executes the complete preprocessing pipeline.
        """

        log_info("=" * 80)
        log_info("Starting Master Dataset Preparation Pipeline")
        log_info("=" * 80)

        ensure_directory_exists(PROCESSED_DATA_DIR)
        ensure_directory_exists(REPORT_DIR)

        self.load_raw_data()

        self.validate_raw_data()

        self.merge_tables()

        self.validate_merge()

        self.select_columns()

        self.engineer_features()

        self.standardize_datatypes()

        self.generate_missing_report()

        self.generate_diagnosis_distribution()

        self.validate_master_dataset()

        self.sample_dataset()

        self.save_artifacts()

        log_info("=" * 80)
        log_info("Master Dataset Successfully Generated")
        log_info("=" * 80)
    
    ####################################################################
    # Loading
    ####################################################################

    def load_raw_data(self) -> None:
        """
        Load the raw MIMIC datasets.

        Raises
        ------
        FileNotFoundError
            If any required dataset is missing.
        """

        log_info("Loading raw datasets...")

        # -----------------------------------------------------------------
        # Verify files exist
        # -----------------------------------------------------------------

        if not PATIENTS_FILE.exists():
            raise FileNotFoundError(f"Patients file not found: {PATIENTS_FILE}")

        if not ADMISSIONS_FILE.exists():
            raise FileNotFoundError(f"Admissions file not found: {ADMISSIONS_FILE}")

        # -----------------------------------------------------------------
        # Read CSV files
        # -----------------------------------------------------------------

        self.patients_df = pd.read_csv(
            PATIENTS_FILE,
            low_memory=False
        )

        self.admissions_df = pd.read_csv(
            ADMISSIONS_FILE,
            low_memory=False
        )

        log_info(
            f"Patients dataset loaded : {self.patients_df.shape}"
        )

        log_info(
            f"Admissions dataset loaded : {self.admissions_df.shape}"
        )

        # -----------------------------------------------------------------
        # Populate summary
        # -----------------------------------------------------------------

        self.summary["patients_rows"] = len(self.patients_df)

        self.summary["admissions_rows"] = len(self.admissions_df)

        self.summary["patients_columns"] = len(self.patients_df.columns)

        self.summary["admissions_columns"] = len(self.admissions_df.columns)

    ####################################################################
    # Raw Validation
    ####################################################################

    def validate_raw_data(self) -> None:
        """
        Validate the raw MIMIC datasets before preprocessing.

        Raises
        ------
        ValueError
            If schema validation fails.
        """

        log_info("Validating raw datasets...")

        REQUIRED_PATIENT_COLUMNS = {
            "subject_id",
            "gender",
            "dob",
        }

        REQUIRED_ADMISSION_COLUMNS = {
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "diagnosis",
        }

        # ------------------------------------------------------------
        # Check required columns
        # ------------------------------------------------------------

        missing_patient_columns = (
            REQUIRED_PATIENT_COLUMNS -
            set(self.patients_df.columns)
        )

        if missing_patient_columns:

            raise ValueError(
                f"Missing PATIENTS columns: {missing_patient_columns}"
            )

        missing_admission_columns = (
            REQUIRED_ADMISSION_COLUMNS -
            set(self.admissions_df.columns)
        )

        if missing_admission_columns:

            raise ValueError(
                f"Missing ADMISSIONS columns: {missing_admission_columns}"
            )

        log_info("Required columns verified.")

        # ------------------------------------------------------------
        # Check empty datasets
        # ------------------------------------------------------------

        if self.patients_df.empty:
            raise ValueError("PATIENTS dataset is empty.")

        if self.admissions_df.empty:
            raise ValueError("ADMISSIONS dataset is empty.")

        log_info("Datasets are non-empty.")

        # ------------------------------------------------------------
        # Duplicate admission IDs
        # ------------------------------------------------------------

        duplicate_hadm = self.admissions_df["hadm_id"].duplicated().sum()

        if duplicate_hadm > 0:

            log_info(
                f"Warning: {duplicate_hadm} duplicated hadm_id values detected."
            )

        else:

            log_info("Admission IDs verified.")

        # ------------------------------------------------------------
        # Datetime parsing check
        # ------------------------------------------------------------

        try:

            pd.to_datetime(self.admissions_df["admittime"])

            pd.to_datetime(self.admissions_df["dischtime"])

            pd.to_datetime(self.patients_df["dob"])

        except Exception:

            raise ValueError(
                "Datetime parsing failed."
            )

        log_info("Datetime columns verified.")

        # ------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------

        self.summary["unique_patients"] = (
            self.patients_df["subject_id"].nunique()
        )

        self.summary["unique_admissions"] = (
            self.admissions_df["hadm_id"].nunique()
        )

        log_info("Raw dataset validation completed.")

    ####################################################################
    # Merge Tables
    ####################################################################

    def merge_tables(self) -> None:
        """
        Merge PATIENTS and ADMISSIONS tables.

        One row in the final dataframe corresponds to one admission.
        """

        log_info("Merging PATIENTS and ADMISSIONS tables...")

        self.master_df = pd.merge(
            self.admissions_df,
            self.patients_df,
            on="subject_id",
            how="left",
            validate="many_to_one"
        )

        log_info(
            f"Merged dataset shape : {self.master_df.shape}"
        )

        self.summary["merged_rows"] = len(self.master_df)
    
    ####################################################################
    # Merge Validation
    ####################################################################

    def validate_merge(self) -> None:
        """
        Validate merged dataset.
        """

        log_info("Validating merged dataset...")

        if self.master_df is None:
            raise ValueError("Merged dataframe not created.")

        # ------------------------------------------------------------
        # Missing patient information
        # ------------------------------------------------------------

        missing_gender = self.master_df["gender"].isna().sum()

        if missing_gender > 0:

            log_info(
                f"Warning : {missing_gender} admissions have missing patient data."
            )

        # ------------------------------------------------------------
        # Verify admission count
        # ------------------------------------------------------------

        if len(self.master_df) != len(self.admissions_df):

            raise ValueError(
                "Merge changed number of admissions."
            )

        log_info("Merge validation completed.")

    ####################################################################
    # Column Selection
    ####################################################################

    def select_columns(self) -> None:
        """
        Select columns required for the master dataset.
        """

        log_info("Selecting required columns...")

        self.master_df = self.master_df[
            MASTER_COLUMNS
        ].copy()

        self.summary["selected_columns"] = len(
            self.master_df.columns
        )

        log_info(
            f"{len(self.master_df.columns)} columns retained."
        )
    
    ####################################################################
    # Feature Engineering
    ####################################################################

    def engineer_features(self) -> None:
        """
        Engineer generic features that are useful for
        downstream prediction tasks.

        Notes
        -----
        Only task-independent features should be created.
        """

        log_info("Engineering generic features...")

        # ------------------------------------------------------------
        # Convert date columns
        # ------------------------------------------------------------

        self.master_df["admittime"] = pd.to_datetime(
            self.master_df["admittime"]
        )

        self.master_df["dischtime"] = pd.to_datetime(
            self.master_df["dischtime"]
        )

        self.master_df["dob"] = pd.to_datetime(
            self.master_df["dob"]
        )

        # ------------------------------------------------------------
        # Age
        # ------------------------------------------------------------

        self.master_df["age"] = (
            (
                self.master_df["admittime"]
                -
                self.master_df["dob"]
            )
            .dt.days
            / 365.25
        ).astype(int)

        # ------------------------------------------------------------
        # Length of Stay
        # ------------------------------------------------------------

        self.master_df["length_of_stay"] = (
            self.master_df["dischtime"]
            -
            self.master_df["admittime"]
        ).dt.total_seconds() / (60 * 60 * 24)

        # ------------------------------------------------------------
        # Admission Year
        # ------------------------------------------------------------

        self.master_df["admission_year"] = (
            self.master_df["admittime"]
            .dt.year
        )

        # ------------------------------------------------------------
        # Admission Month
        # ------------------------------------------------------------

        self.master_df["admission_month"] = (
            self.master_df["admittime"]
            .dt.month
        )

        # ------------------------------------------------------------
        # Admission Weekday
        # ------------------------------------------------------------

        self.master_df["admission_weekday"] = (
            self.master_df["admittime"]
            .dt.day_name()
        )

        self.summary["engineered_features"] = [

            "age",

            "length_of_stay",

            "admission_year",

            "admission_month",

            "admission_weekday"

        ]

        log_info(
            "Created 5 engineered features."
        )
    
    ####################################################################
    # Datatype Standardization
    ####################################################################

    def standardize_datatypes(self) -> None:
        """
        Standardize dataframe datatypes.
        """

        log_info("Standardizing datatypes...")

        categorical_columns = [

            "gender",

            "admission_type",

            "admission_location",

            "discharge_location",

            "insurance",

            "language",

            "religion",

            "marital_status",

            "ethnicity",

            "diagnosis",

            "admission_weekday"

        ]

        for column in categorical_columns:

            if column in self.master_df.columns:

                self.master_df[column] = (
                    self.master_df[column]
                    .astype("category")
                )

        log_info("Datatype standardization completed.")

        self.summary["feature_count"] = len(
        self.master_df.columns
    )

        self.summary["categorical_features"] = categorical_columns
    
    ####################################################################
    # Missing Value Report
    ####################################################################

    def generate_missing_report(self) -> None:
        """
        Generate missing value report for the master dataset.
        """

        log_info("Generating missing value report...")

        missing_report = pd.DataFrame({

            "column": self.master_df.columns,

            "missing_count": self.master_df.isna().sum().values,

            "missing_percentage":

                (
                    self.master_df.isna().mean() * 100
                ).round(2).values

        })

        missing_report = missing_report.sort_values(

            by="missing_percentage",

            ascending=False

        )

        missing_report.to_csv(

            MISSING_VALUE_REPORT_FILE,

            index=False

        )

        self.summary["missing_report"] = str(

            MISSING_VALUE_REPORT_FILE

        )

        log_info("Missing value report generated.")
    
    ####################################################################
    # Diagnosis Distribution
    ####################################################################

    def generate_diagnosis_distribution(self) -> None:
        """
        Generate diagnosis frequency distribution.
        """

        log_info("Generating diagnosis distribution...")

        diagnosis_distribution = (

            self.master_df["diagnosis"]

            .value_counts()

            .reset_index()

        )

        diagnosis_distribution.columns = [

            "diagnosis",

            "count"

        ]

        diagnosis_distribution["percentage"] = (

            diagnosis_distribution["count"]

            /

            len(self.master_df)

            *

            100

        ).round(2)

        diagnosis_distribution.to_csv(

            DIAGNOSIS_DISTRIBUTION_FILE,

            index=False

        )

        self.summary["unique_diagnoses"] = len(

            diagnosis_distribution

        )

        log_info(

            f"Found {len(diagnosis_distribution)} unique diagnoses."
        )
    
    ####################################################################
    # Dataset Profile
    ####################################################################

    def generate_dataset_profile(self) -> None:
        """
        Generate structural metadata about the dataset.
        """

        log_info("Generating dataset profile...")

        profile = {

            "total_rows":

                len(self.master_df),

            "total_columns":

                len(self.master_df.columns),

            "identifier_columns":[

                "subject_id",

                "hadm_id"

            ],

            "categorical_columns":[

                column

                for column in self.master_df.columns

                if str(self.master_df[column].dtype) == "category"

            ],

            "numerical_columns":[

                column

                for column in self.master_df.columns

                if self.master_df[column].dtype

                in ["int64","float64"]

            ],

            "datetime_columns":[

                column

                for column in self.master_df.columns

                if "datetime"

                in str(self.master_df[column].dtype)

            ],

            "engineered_features":

                self.summary["engineered_features"]

        }

        save_json(

            profile,

            DATASET_PROFILE_FILE

        )

        self.summary["dataset_profile"] = str(

            DATASET_PROFILE_FILE

        )

        log_info("Dataset profile generated.")
    
    ####################################################################
    # Master Dataset Validation
    ####################################################################

    def validate_master_dataset(self) -> None:
        """
        Validate the processed master dataset.
        """

        log_info("Validating master dataset...")

        if self.master_df is None:
            raise ValueError("Master dataframe has not been created.")

        # ------------------------------------------------------------
        # Check duplicate admissions
        # ------------------------------------------------------------

        duplicate_admissions = self.master_df["hadm_id"].duplicated().sum()

        if duplicate_admissions > 0:
            raise ValueError(
                f"Found {duplicate_admissions} duplicate hadm_id values."
            )

        # ------------------------------------------------------------
        # Check negative length of stay
        # ------------------------------------------------------------

        negative_los = (self.master_df["length_of_stay"] < 0).sum()

        if negative_los > 0:
            raise ValueError(
                f"Found {negative_los} negative Length of Stay values."
            )

        # ------------------------------------------------------------
        # Check age
        # ------------------------------------------------------------

        negative_age = (self.master_df["age"] < 0).sum()

        if negative_age > 0:
            raise ValueError(
                f"Found {negative_age} negative ages."
            )

        # ------------------------------------------------------------
        # MIMIC de-identification warning
        # ------------------------------------------------------------

        very_old = (self.master_df["age"] > 120).sum()

        if very_old > 0:
            log_info(
                f"{very_old} patients have age >120 "
                "(expected in MIMIC due to de-identification)."
            )

        log_info("Master dataset validation completed.")
    

    ####################################################################
    # Dataset Sampling
    ####################################################################

    def sample_dataset(self) -> None:
        """
        Sample the master dataset for the POC.
        """

        log_info("Sampling dataset...")

        original_size = len(self.master_df)

        if original_size > SAMPLE_SIZE:

            self.master_df = (
                self.master_df
                .sample(
                    n=SAMPLE_SIZE,
                    random_state=RANDOM_STATE
                )
                .reset_index(drop=True)
            )

            log_info(
                f"Sampled {SAMPLE_SIZE} admissions "
                f"from {original_size} admissions."
            )

        else:

            log_info(
                "Dataset size is already below sampling threshold."
            )

        self.summary["sample_size"] = len(self.master_df)
        self.summary["original_size"] = original_size
    
    ####################################################################
    # Save Outputs
    ####################################################################

    def save_artifacts(self) -> None:
        """
        Save all generated artifacts.
        """

        log_info("Saving artifacts...")

        # ------------------------------------------------------------
        # Save Master Dataset
        # ------------------------------------------------------------

        self.master_df.to_csv(
            MASTER_DATASET_FILE,
            index=False
        )

        # ------------------------------------------------------------
        # Dataset Summary
        # ------------------------------------------------------------

        self.summary["pipeline_version"] = "1.0"

        self.summary["dataset_name"] = "Master Dataset"

        self.summary["dataset_source"] = "MIMIC-III"

        self.summary["created_at"] = current_timestamp()

        self.summary["total_rows"] = len(self.master_df)

        self.summary["total_columns"] = len(self.master_df.columns)

        save_json(
            self.summary,
            DATASET_SUMMARY_FILE
        )

        log_info("Artifacts successfully saved.")
