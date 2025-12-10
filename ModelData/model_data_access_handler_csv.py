

from dataclasses import dataclass
import json

import pandas as pd
from ModelData.i_model_data_access_handler import IModelDataAccessHandler
import os
import requests
import zipfile
from tqdm import tqdm
import numpy as np
import datetime as dt
from ModelTypes.ais_dataset_raw import AISDatasetRaw
from ForceTypes.vessel_types import VesselType
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from shapely.geometry import shape, Point


@dataclass
class Config:
    area: str  # ????????????????? needs to be polygon
    num_workers: int
    output_dir: str = "Data"


class ModelDataAccessHandlerCSV(IModelDataAccessHandler):
    def __init__(self, config: Config):
        self._cfg = config
        self._cols_to_use = {
            "Timestamp": True,
            "Type of mobile": False,
            "MMSI": True,
            "Latitude": True,
            "Longitude": True,
            "Navigational status": False,
            "ROT": True,
            "SOG": True,
            "COG": True,
            "Heading": True,
            "IMO": False,
            "Callsign": False,
            "Name": False,
            "Ship type": True,
            "Cargo type": False,
            "Width": False,
            "Length": False,
            "Type of position fixing device": False,
            "Draught": True,
            "Destination": False,
            "ETA": False,
            "Data source type": False,
            "A": False,
            "B": False,
            "C": False,
            "D": False,
        }

   # Download raw data of AISDK and AISUS
    def _download_ais_dataset(self, file_name: str):
        raw_data_path = "Data/"
        # Step 1: Set base information about raw dataset
        download_url = "http://aisdata.ais.dk/2024/"
        csv_file_name = f"{file_name}.csv"
        print(csv_file_name)

        # Step 2: Check if CSV file already exists
        csv_file_path = os.path.join(raw_data_path, "AISDataRaw", csv_file_name)

        if os.path.exists(csv_file_path):
            print(f"CSV file '{csv_file_path}' already exists. No download needed.")
            return csv_file_path

        # Step 3: Download and unzip if CSV doesn't exist
        def attempt_download(url, zip_path):
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                total_length = int(response.headers.get('content-length', 0))
                with open(zip_path, 'wb') as file, tqdm(
                    desc=zip_path,
                    total=total_length,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=8192):
                        size = file.write(chunk)
                        bar.update(size)
                print(f"ZIP file downloaded successfully as {zip_path}")
                return True
            except requests.exceptions.HTTPError:
                return False

        # First attempt
        if not os.path.exists(raw_data_path):
            os.makedirs(raw_data_path)
        zip_path = os.path.join(raw_data_path, "AISDataRaw", f"{file_name}.zip")
        url = download_url + file_name + ".zip"
        if not attempt_download(url, zip_path):
            # Second attempt
            url = download_url + file_name[:-3] + ".zip"
            zip_path = os.path.join(raw_data_path, "AISDataRaw", f"{file_name[:-3]}.zip")
            if not attempt_download(url, zip_path):
                print(f"Error: Unable to download the file for {file_name}. The file may not exist.")
                return None

        def unzip_file(zip_path, extract_to):
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
                    print(f"File '{zip_path}' has been unzipped.")
            except zipfile.BadZipFile:
                print(f"Error: The file '{zip_path}' is not a valid ZIP file.")
            except Exception as e:
                print(f"Error unzipping the file '{zip_path}': {e}")
        # Unzip the file
        unzip_file(zip_path, os.path.join(raw_data_path, "AISDataRaw"))

        # Check if CSV file now exists after unzipping
        if not os.path.exists(csv_file_path):
            print(f"Error: CSV file '{csv_file_path}' not found after unzipping.")
            return None

        return csv_file_path

    def download_csv_files(self, dates: list[dt.date]) -> list[str]:
        # Collect list of file names to download
        file_names: list[str] = []

        for cur_date in dates:
            if cur_date.month in [1, 2] and cur_date.year == 2024:
                raise ValueError(f"Date {cur_date} is not available for download.")

            file_name = f"aisdk-{cur_date.year}-{cur_date.month:02d}-{cur_date.day:02d}"

            file_names.append(file_name)
            cur_date += dt.timedelta(days=1)

        # Parallel download
        csv_paths: list[str] = []
        num_workers = getattr(self._cfg, "num_workers", 4)

        print(f"Downloading {len(file_names)} AIS files using {num_workers} threads...")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_map = {
                executor.submit(self._download_ais_dataset, fn): fn
                for fn in file_names
            }

            for future in as_completed(future_map):
                fname = future_map[future]
                try:
                    result = future.result()
                    if result:
                        csv_paths.append(result)
                except Exception as e:
                    print(f"Download failed for {fname}: {e}")

        return csv_paths

    def _col_indexes_to_use(self):
        # Return indices of True columns
        return [i for i, (col, use) in enumerate(self._cols_to_use.items()) if use]

    def _col_name_to_index(self, col_name: str):
        """
        Given a column name, return the index it occupies among the
        True-valued fields in self.cols_to_use.
        """
        if col_name not in self._cols_to_use:
            raise KeyError(f"Column '{col_name}' does not exist in cols_to_use")

        # Build the list of columns actually used (True values)
        true_cols = [name for name, use in self._cols_to_use.items() if use]

        try:
            return true_cols.index(col_name)
        except ValueError:
            raise ValueError(f"Column '{col_name}' is not enabled (set to False)")

    def _parse_vessel_type(self, raw_value: str) -> float:
        if not raw_value:
            return float(VesselType.UNKNOWN.value)

        key = re.sub(r"[\/\-\s]", "", raw_value).upper()

        enum_val = VesselType.__members__.get(key, VesselType.UNKNOWN)
        return float(enum_val.value)

    def _coarse_processed_data_filename(self, csv_file_name: str) -> str:
        fields: str = "_".join([f"{k}" for k, v in self._cols_to_use.items() if v])
        return f"{self._cfg.output_dir}/AISDatasetRaw/{os.path.basename(csv_file_name).replace('.csv', '')}_cols_{fields}.npz"

    def get_ais_messages(self, date: dt.date) -> AISDatasetRaw:
        csv_file_path = self.download_csv_files([date])[0]
        np_file_path = self._coarse_processed_data_filename(csv_file_path)

        with open("Data/assets/eez.json", "r") as f:
            zone = json.load(f)

        if os.path.exists(np_file_path):
            print(f"{np_file_path} np file exists - skipping...")
            return AISDatasetRaw.load(np_file_path)

        print("Loading CSV file into DataFrame...")
        df = pd.read_csv(
            csv_file_path,
            delimiter=",",
            dtype=str,
            on_bad_lines='warn',  # or 'skip' if you want silent handling
            usecols=self._col_indexes_to_use(),
            header=0
        )

        # --- Cleaning phase ---
        print("Filtering unwanted rows...")
        invalid_values = ["Unknown", "Undefined", "None", "", "nan", "NaN", "N/A", "NULL"]
        df = df[~df.isin(invalid_values).any(axis=1)]

        # --- Parsing phase ---
        print("Converting vessel type to numeric...")
        ship_type_col = "Ship type"
        df[ship_type_col] = df[ship_type_col].apply(self._parse_vessel_type)

        print("Converting timestamps to UNIX time...")
        ts_col = "# Timestamp"
        df[ts_col] = pd.to_datetime(df[ts_col], format="%d/%m/%Y %H:%M:%S", errors="coerce")
        df = df.dropna(subset=[ts_col])
        df[ts_col] = df[ts_col].map(lambda x: x.timestamp())

        print("Converting data to floats...")
        for col in df.columns:
            if col != ts_col:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        print("Dropping rows with NaN values...")
        df = df.dropna()

        print("Filtering cargos and tankers...")
        df = df[(df["Ship type"] == VesselType.CARGO.value) | (df["Ship type"] == VesselType.TANKER.value)]

        print("Filtering out-of-bounds latitude and longitude...")
        df = df[((df['Latitude'] >= -90) & (df['Latitude'] <= 90)) &
                ((df['Longitude'] >= -180) & (df['Longitude'] <= 180))]

        print("Removing duplicate messages (same MMSI + timestamp)...")
        df = df.drop_duplicates(subset=["MMSI", "# Timestamp"])

        print("Remove data outside area of interest...")
        multipolygon = shape(zone["features"][0]["geometry"])
        points = [Point(lon, lat) for lat, lon in zip(df['Latitude'], df['Longitude'])]
        mask = []
        for point in tqdm(points, desc="Filtering points by area"):
            mask.append(multipolygon.contains(point))

        df = df[mask]

        print("Converting DataFrame to numpy array...")
        data_array = df.to_numpy(dtype=float)
        dataset = AISDatasetRaw(data_array)

        print("NaN count per column:\n", df.isna().sum())
        print("Any inf values:", np.isinf(df.to_numpy()).any())

        assert not np.isnan(data_array).any(), "Data contains NaN values after processing."
        assert not np.isinf(data_array).any(), "Data contains infinite values after processing."

        print("Saving filtered data to npy file...")
        os.makedirs(os.path.dirname(np_file_path), exist_ok=True)
        dataset.save(np_file_path)

        return dataset
