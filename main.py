from DataAccess.i_data_access_handler import AreaTuple
from ForceProviders.traffic_force_provider import (
    TrafficForceProvider,
    Config as TFPConfig,
)
import datetime as dt
from Types.latlon import LatLon
from Utils.heatmap_generator import (
    Config as HGCfg,
    generate_heatmap_image,
)
from DataAccess.mock_data_access_handler import MockDataAccessHandler
from DataAccess.postgres_connection import PostgresConnection, Config as PGCfg


def main():
    tcfg = TFPConfig(
        start_date=dt.date(2024, 5, 4),
        end_date=dt.date(2024, 5, 6),
        sample_rate=1,
        area=AreaTuple(
            LatLon(54.622382, 7.269879),
            LatLon(57.750526, 12.868355)
            # LatLon(56.188642, 9.959005),
            # LatLon(57.561181, 12.011192)
        ),
        vessel_types=[],
        base_zoom=22,
        active_zoom=19
    )

    prov = TrafficForceProvider(tcfg, data_handler=MockDataAccessHandler("Data/aisdk-2024-05-05.csv"))

    cfg = HGCfg(
        vectormap=prov._vectormap
    )

    generate_heatmap_image(cfg)


if __name__ == "__main__":
    main()
