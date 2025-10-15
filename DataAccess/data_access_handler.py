from tqdm import tqdm
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

        with tqdm(dates, desc="Fetching data for dates") as pbar:
            for date in pbar:
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

                # Update progress bar with number of records fetched so far
                pbar.set_postfix({"records_so_far": len(all_results)})

        print(f"Finished fetching data. Total records: {len(all_results)}")
        return all_results

    def get_ais_messages_no_stops(self, dates: list[datetime.date], area: AreaTuple) -> list[AisMessageTuple]:
        all_results = []

        query_template = """
        WITH ships AS (
            SELECT DISTINCT vessel_id
            FROM dim.vessel_dim
            WHERE LENGTH(mmsi::text) = 9
            AND LEFT(mmsi::text, 1) BETWEEN '2' AND '7'
        )
        SELECT cur.lat, cur.lon
        FROM fact.ais_point_fact cur
        JOIN fact.ais_point_fact prev
            ON cur.prev_ais_point_id = prev.ais_point_id
        JOIN dim.date_dim dd
            ON cur.date_id = dd.date_id AND prev.date_id = dd.date_id
        JOIN dim.time_dim cur_td
            ON cur.time_id = cur_td.time_id
        JOIN dim.time_dim prev_td
            ON prev.time_id = prev_td.time_id
        JOIN ships s
            ON cur.vessel_id = s.vessel_id
        WHERE dd.year_no = %s AND dd.month_no = %s AND dd.day_no = %s
        AND cur.lat BETWEEN %s AND %s
        AND cur.lon BETWEEN %s AND %s
        AND (
            cur.sog > %s
            OR cur_td.hour_no - prev_td.hour_no > %s
            OR ST_DistanceSphere(
                ST_MakePoint(prev.lon, prev.lat),
                ST_MakePoint(cur.lon, cur.lat)
            ) > %s
        );
        """

        print(f"Fetching AIS data for {len(dates)} days in area {area}...")

        speed_threshold = 1  # knots
        time_threshold = 1.5  # hours
        distance_threshold = 2000  # meters

        with tqdm(dates, desc="Fetching data for dates") as pbar:
            for date in pbar:
                params = (
                    date.year,
                    date.month,
                    date.day,
                    area.bot_left.lat, area.top_right.lat,
                    area.bot_left.lon, area.top_right.lon,
                    speed_threshold,
                    time_threshold,
                    distance_threshold
                )

                day_results = self.db_connection.execute_query(query_template, params)

                if day_results:
                    all_results.extend(day_results)

                # Update progress bar with number of records fetched so far
                pbar.set_postfix({"records_so_far": len(all_results)})

        print(f"Finished fetching data. Total records: {len(all_results)}")
        return all_results
