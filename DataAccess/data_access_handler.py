from DataAccess.i_data_connection import IDataConnection
from DataAccess.i_data_access_handler import IDataAccessHandler, AisMessageTuple, AreaTuple
import datetime


class DataAccessHandler(IDataAccessHandler):
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
                area.bot_left.lat, area.top_right.lat,
                area.bot_left.lon, area.top_right.lon
            )

            day_results = self.db_conn.execute_query(query_template, params)

            if day_results:
                all_results.extend(day_results)

        print(f"Finished fetching data. Total records: {len(all_results)}")
        return all_results
