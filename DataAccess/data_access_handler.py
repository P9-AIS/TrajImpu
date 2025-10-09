from DataAccess.i_data_connection import IDataConnection
from DataAccess.i_data_access_handler import IDataAccessHandler, AisMessageTuple, AreaTuple
import datetime


class DataAccessHandler(IDataAccessHandler):
    def __init__(self, db_connection: IDataConnection):
        self.db_connection = db_connection

    def get_ais_messages(self, dates: list[datetime.date], area: AreaTuple) -> list[AisMessageTuple]:
        all_results = []

        query_template = """
        SELECT lat, lon
        FROM fact.ais_point_fact
        JOIN dim.date_dim dd
            ON ais_point_fact.date_id = dd.date_id
        WHERE dd.year_no = %s AND dd.month_no = %s AND dd.day_no = %s
        AND lat BETWEEN %s AND %s
        AND lon BETWEEN %s AND %s;
        """

        print(f"Fetching AIS data for {len(dates)} days in area {area}...")

        for date in dates:
            params = (
                date.year,
                date.month,
                date.day,
                area.bot_left.lat, area.top_right.lat,
                area.bot_left.lon, area.top_right.lon
            )

            day_results = self.db_connection.execute_query(query_template, params)

            if day_results:
                all_results.extend(day_results)

        print(f"Finished fetching data. Total records: {len(all_results)}")
        return all_results
