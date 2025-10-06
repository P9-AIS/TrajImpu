from DataAccess.i_data_access_handler import AreaTuple
from ForceProviders.traffic_force_provider import (
    TrafficForceProvider,
    Config as TFPConfig,
)
import datetime as dt
from Types.latlon import LatLon
from params import Params
from Utils.heatmap_generator import (
    Config as HGCfg,
    generate_vectormap,
    traffic_force_field,
    generate_heatmap,
)
from DataAccess.mock_data_access_handler import MockDataAccessHandler
from DataAccess.postgres_connection import PostgresConnection, Config as PGCfg


def main():
    tcfg = TFPConfig(
        start_date=dt.date(2025, 3, 1),
        end_date=dt.date(2025, 3, 3),
        sample_rate=1,
        area=AreaTuple(
            LatLon(56.188642, 9.959005),
            LatLon(57.561181, 12.011192)
        ),
        vessel_types=[],
        base_zoom=22,
        active_zoom=21
    )

    prov = TrafficForceProvider(tcfg, data_handler=MockDataAccessHandler("Data/aisdk-2025-03-02.csv"))

    cfg = HGCfg(
        area=AreaTuple(
            LatLon(56.188642, 9.959005),
            LatLon(57.561181, 12.011192)
        ),
        zoom_level=21
    )
    force_field = traffic_force_field(prov, cfg)

    generate_heatmap(force_field, tile_size=5)
    generate_vectormap(force_field)


if __name__ == "__main__":
    main()
