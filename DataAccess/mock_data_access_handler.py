import csv
from DataAccess.i_data_access_handler import (
    IDataAccessHandler,
    AisMessageTuple,
    AreaTuple,
)
import datetime
# Import the tqdm library
from tqdm import tqdm


class MockDataAccessHandler(IDataAccessHandler):
    def __init__(self, csv_file: str):
        self.csv_file = csv_file

    def get_ais_messages(
        self, dates: list[datetime.date], area: AreaTuple
    ) -> list[AisMessageTuple]:
        results: list[AisMessageTuple] = []

        # Convert the list of dates to a set of date strings (YYYY-MM-DD) for fast lookup
        target_dates = {d.strftime("%Y-%m-%d") for d in dates}

        try:
            # Count the total lines in the file to provide a total for the progress bar
            with open(self.csv_file, mode="r", newline="", encoding="utf-8") as f:
                # Use a generator expression for efficiency
                total_rows = sum(1 for _ in f) - 1  # Subtract 1 for the header

            with open(self.csv_file, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)

                # Skip header row
                next(reader, None)

                # Wrap the reader iterator with tqdm to display a progress bar
                for row in tqdm(reader, total=total_rows, desc="Processing AIS data"):
                    try:
                        # 1. Type Conversion and Parsing
                        (
                            ts_str,
                            _,
                            _,
                            lat_str,
                            lon_str,
                            _,
                            _,
                            sog_str,
                            cog_str,
                            _,
                            _,
                            _,
                            _,
                            vessel_type,
                            _,
                            _,
                            _,
                            _,
                            _,
                            _,
                            _,
                            _,
                            _,
                            _,
                            _,
                            _,
                        ) = row

                        # Assume ISO format for timestamp
                        timestamp = datetime.datetime.strptime(ts_str, "%d/%m/%Y %H:%M:%S")
                        latitude = float(lat_str)
                        longitude = float(lon_str)
                        sog = float(sog_str or 0)
                        cog = float(cog_str or 0)
                        # vessel_type is kept as string

                        # 2. Filtering by Date
                        # Extract the date part of the timestamp for fast set comparison
                        row_date_str = timestamp.strftime("%Y-%m-%d")
                        if row_date_str not in target_dates:
                            continue

                        # 3. Filtering by Area (Geographical Bounds)
                        if not (
                            area.bot_left.lat <= latitude <= area.top_right.lat
                            and area.bot_left.lon <= longitude <= area.top_right.lon
                        ):
                            continue

                        # 4. Create and store the AisMessageTuple
                        message = AisMessageTuple(
                            timestamp=timestamp,
                            latitude=latitude,
                            longitude=longitude,
                            sog=sog,
                            cog=cog,
                            vessel_type=vessel_type,
                        )
                        results.append(message)

                    except ValueError:
                        # Skip rows with invalid data types (e.g., non-numeric)
                        continue

        except FileNotFoundError:
            print(f"Error: CSV file not found at {self.csv_file}")
            return []
        except Exception as e:
            print(f"An unexpected error occurred while reading the CSV: {e}")
            return []

        print(f"Successfully processed CSV. Found {len(results)} filtered records.")
        return results
