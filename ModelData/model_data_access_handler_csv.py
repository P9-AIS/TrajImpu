

from ModelData.i_model_data_access_handler import IModelDataAccessHandler
import os
import requests
import zipfile
from tqdm import tqdm
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import datetime as dt
from ModelData.i_model_data_access_handler import AisMessageTuple
from Types.vessel_types import VesselType
from Types.area import Area
from ModelData.i_model_data_access_handler import Config
from concurrent.futures import ThreadPoolExecutor, as_completed


class ModelDataAccessHandlerCSV(IModelDataAccessHandler):
    def __init__(self, config: Config):
        self.config = config
        self.cols_to_use = {
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
    def download_ais_dataset(self, file_name: str):
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
        unzip_file(zip_path, os.path.join(raw_data_path, "csv_files"))

        # Check if CSV file now exists after unzipping
        if not os.path.exists(csv_file_path):
            print(f"Error: CSV file '{csv_file_path}' not found after unzipping.")
            return None

        return csv_file_path

    def download_csv_files(self) -> list[str]:
        # Collect list of file names to download
        cur_date: dt.date = self.config.date_start
        file_names: list[str] = []

        while cur_date <= self.config.date_end:
            if cur_date == dt.date(2024, 1, 1) or cur_date == dt.date(2024, 2, 1):
                file_name = f"aisdk-{cur_date.year}-{cur_date.month:02d}"
            elif dt.date(2024, 1, 2) <= cur_date <= dt.date(2024, 1, 31) or \
                    dt.date(2024, 2, 2) <= cur_date <= dt.date(2024, 2, 29):
                cur_date += dt.timedelta(days=1)
                continue
            else:
                file_name = f"aisdk-{cur_date.year}-{cur_date.month:02d}-{cur_date.day:02d}"

            file_names.append(file_name)
            cur_date += dt.timedelta(days=1)

        # Parallel download
        csv_paths: list[str] = []
        num_workers = getattr(self.config, "num_workers", 4)

        print(f"Downloading {len(file_names)} AIS files using {num_workers} threads...")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_map = {
                executor.submit(self.download_ais_dataset, fn): fn
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
        return [i for i, (col, use) in enumerate(self.cols_to_use.items()) if use]

    def _col_name_to_index(self, col_name: str):
        """
        Given a column name, return the index it occupies among the
        True-valued fields in self.cols_to_use.
        """
        if col_name not in self.cols_to_use:
            raise KeyError(f"Column '{col_name}' does not exist in cols_to_use")

        # Build the list of columns actually used (True values)
        true_cols = [name for name, use in self.cols_to_use.items() if use]

        try:
            return true_cols.index(col_name)
        except ValueError:
            raise ValueError(f"Column '{col_name}' is not enabled (set to False)")

    def _parse_vessel_type(self, raw_value: str) -> float:
        if not raw_value:
            return float(VesselType.UNKNOWN.value)

        key = (
            raw_value.replace("/", "")
            .replace("-", "")
            .replace(" ", "")
            .upper()
        )

        enum_val = VesselType.__members__.get(key, VesselType.UNKNOWN)
        return float(enum_val.value)

    def _coarse_processed_data_filename(self, csv_file_name: str) -> str:
        fields: str = "_".join([f"{k}" for k, v in self.cols_to_use.items() if v])
        return f"Data/np_files/{os.path.basename(csv_file_name).replace('.csv', '')}_cols_{fields}.npz"

    def get_ais_messages(self) -> list[AisMessageTuple]:

        csv_file_paths = self.download_csv_files()
        np_file_paths: list[str] = []

        for csv_file_name in tqdm(csv_file_paths, desc="Processing CSV files"):
            np_filename = self._coarse_processed_data_filename(csv_file_name)
            np_file_paths.append(np_filename)

            if os.path.exists(np_filename):
                print(f"{np_filename} np file exists - skipping...")
                continue
                # file_data = np.load(np_filename)
                # all_filtered_data = np.vstack((all_filtered_data, file_data['filtered_data']))
            else:
                print("Loading csv file to np...")
                raw_data = np.genfromtxt(
                    csv_file_name,
                    delimiter=",",
                    dtype=str,
                    skip_header=1,
                    usecols=self._col_indexes_to_use()
                )

                # filter unwanted rows
                mask = ~np.isin(raw_data, ["Unknown", "Undefined", "None", ""]).any(axis=1)
                filtered_data = raw_data[mask]

                # parse vessel type column
                print("Converting vessel type to numeric...")
                vt_idx = self._col_name_to_index("Ship type")
                filtered_data[:, vt_idx] = [
                    self._parse_vessel_type(v)
                    for v in filtered_data[:, vt_idx]
                ]

                print("Converting timestamp to unix time...")
                # parse datetime
                time_idx = self._col_name_to_index("Timestamp")
                filtered_data[:, time_idx] = [
                    dt.datetime.strptime(v, "%d/%m/%Y %H:%M:%S").timestamp()
                    for v in filtered_data[:, time_idx]
                ]

                # convert to float
                print("Converting data to floats...")
                filtered_data = filtered_data.astype(float)

                # save
                print("Saving filtered data to npy file...")
                os.makedirs(os.path.dirname(np_filename), exist_ok=True)
                np.savez_compressed(np_filename, filtered_data=filtered_data)

        print("Combining all filtered data...")
        all_filtered_data = np.empty((0, len(self._col_indexes_to_use())), dtype=float)
        for np_file in tqdm(np_file_paths, desc="Combining np files"):
            np_data = np.load(np_file)
            if np_data:
                all_filtered_data = np.vstack((all_filtered_data, np_data['filtered_data']))

        print("Converting filtered data to AisMessageTuple list...")
        all_messages = list(AisMessageTuple(*row) for row in all_filtered_data)
        return all_messages
