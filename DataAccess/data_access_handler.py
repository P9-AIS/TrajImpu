from collections import namedtuple
from DataAccess.i_data_connection import IDataConnection
import datetime

AisMessageTuple = namedtuple('AisMessageTuple', ['timestamp', 'latitude', 'longitude', 'sog', 'cog', 'vessel_type'])
AreaTuple = namedtuple('AreaTuple', ['lat_min', 'lon_min', 'lat_max', 'lon_max'])


class DataAccessHandler:
    def __init__(self, db_connection: IDataConnection):
        self.db_connection = db_connection

    def get_ais_messages(self, dates: list[datetime.date], area: AreaTuple) -> list[AisMessageTuple]:
        all_results = []

        query_template = """
        SELECT timestamp, latitude, longitude, sog, cog, vessel_type
        FROM ais_data
        WHERE timestamp >= %s AND timestamp < %s
          AND latitude BETWEEN %s AND %s
          AND longitude BETWEEN %s AND %s
        """

        print(f"Fetching AIS data for {len(dates)} days in area {area}...")

        for date in dates:
            start_of_day = date
            end_of_day = start_of_day + datetime.timedelta(days=1)

            params = (
                start_of_day,
                end_of_day,
                area.lat_min, area.lat_max,
                area.lon_min, area.lon_max
            )

            day_results = self.db_conn.execute_query(query_template, params)

            if day_results:
                all_results.extend(day_results)

        print(f"Finished fetching data. Total records: {len(all_results)}")
        return all_results
