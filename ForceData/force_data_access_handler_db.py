from tqdm import tqdm
from Connection.i_data_connection import IDataConnection
from ForceData.i_force_data_access_handler import IForceDataAccessHandler, AisMessageTuple, DepthTuple
import datetime
from ForceTypes.area import Area
from ForceTypes.espg3034_coord import Espg3034Coord
from ForceUtils.geo_converter import GeoConverter as gc

dk_espg3032_bound_top_left = Espg3034Coord(3592900.0, 3475350.0)


class ForceDataAccessHandlerDb(IForceDataAccessHandler):
    def __init__(self, db_connection: IDataConnection):
        self.db_connection = db_connection

    def get_ais_messages_no_stops(self, dates: list[datetime.date], area: Area) -> list[AisMessageTuple]:
        all_results = []

        query = """
        WITH ships AS (
            SELECT DISTINCT vessel_id
            FROM dim.vessel_dim
            WHERE LENGTH(mmsi::text) = 9
            AND LEFT(mmsi::text, 1) BETWEEN '2' AND '7'
        )
        SELECT cur.lon, cur.lat
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
        AND cur.lon BETWEEN -20 AND 40
        AND cur.lat BETWEEN 30 AND 80
        AND st_contains(
            st_geomfromtext(
                %s,
                3034
            ), 
            ST_Transform(
                ST_SetSRID(ST_MakePoint(cur.lon, cur.lat), 4326),
                3034
            )
        )
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

        polygon_wkt = \
            f"POLYGON(({area.bottom_left.E} {area.bottom_left.N}, " + \
            f"{area.bottom_left.E} {area.top_right.N}, " + \
            f"{area.top_right.E} {area.top_right.N}, " + \
            f"{area.top_right.E} {area.bottom_left.N}, " + \
            f"{area.bottom_left.E} {area.bottom_left.N}))"

        speed_threshold = 1  # knots
        time_threshold = 1.5  # hours
        distance_threshold = 2000  # meters

        with tqdm(dates, desc="Fetching data for dates") as pbar:
            for date in pbar:
                params = (
                    date.year,
                    date.month,
                    date.day,
                    polygon_wkt,
                    speed_threshold,
                    time_threshold,
                    distance_threshold
                )

                day_results = self.db_connection.execute_query(query, params)

                if day_results:
                    all_results.extend(day_results)

                pbar.set_postfix({"records_so_far": len(all_results)})

        print(f"Finished fetching data. Total records: {len(all_results)}")
        return all_results

    def get_depths(self, area: Area) -> tuple[int, list[DepthTuple]]:
        # query = """
        # SELECT x, y, depth
        # FROM dim.gst_depth_grid_dim
        # WHERE st_contains(
        #     st_geomfromtext(
        #         %s,
        #         3034
        #     ),
        #     geom
        # );

        # """

        # polygon_wkt = \
        #     f"POLYGON(({area.bottom_left.E} {area.bottom_left.N}, " + \
        #     f"{area.bottom_left.E} {area.top_right.N}, " + \
        #     f"{area.top_right.E} {area.top_right.N}, " + \
        #     f"{area.top_right.E} {area.bottom_left.N}, " + \
        #     f"{area.bottom_left.E} {area.bottom_left.N}))"

        # params = (
        #     polygon_wkt,
        # )

        # print(f"Fetching depth data...")

        # results = self.db_connection.execute_query(query, params)

        # load form pickle
        import pickle
        with open('depth_data.pkl', 'rb') as f:
            results = pickle.load(f)

        print(f"Fetched {len(results)} depth records.")

        processed_results = []

        dk_top_left_x, dk_top_left_y = gc.epsg3034_to_cell(
            dk_espg3032_bound_top_left.E, dk_espg3032_bound_top_left.N, 25, 25)

        for (x_small, y_small, depth) in tqdm(results, desc="Processing depth records"):
            x_big = x_small + dk_top_left_x
            y_big = dk_top_left_y - y_small

            E, N = gc.cell_to_epsg3034(x_big, y_big, 25, 25)

            processed_results.append(DepthTuple(E, N, depth))

        return 50, processed_results
