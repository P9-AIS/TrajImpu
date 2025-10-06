from ForceProviders.traffic_force_provider import TrafficForceProvider, Config as TFPConfig
import datetime as dt
from params import Params
from Utils.heatmap_generator import Config as HGCfg, generate_vectormap, traffic_force_field, generate_heatmap
from DataAccess.data_access_handler import DataAccessHandler
from DataAccess.postgres_connection import PostgresConnection, Config as PGCfg


def main():
    prov = TrafficForceProvider(TFPConfig(
        dt.date(2025, 10, 10),
        dt.date(2025, 10, 10),
        2,
        57.813925, 7.544589,
        54.624055, 15.958911,
        [],
        18,
        10
    ),
        data_handler=DataAccessHandler(PostgresConnection(PGCfg(
            "localhost",
            5432,
            "postgres",
            "postgres",
            "password"
        )))
    )

    # print(prov.get_force(Params(
    #     0,
    #     57.066227, 9.790968,
    #     3,
    #     3,
    #     3
    # )
    # ))

    cfg = HGCfg(
        57.813925, 7.544589,
        54.624055, 15.958911,
        10)

    force_field = traffic_force_field(prov, cfg)
    generate_heatmap(force_field)
    generate_vectormap(force_field)


if __name__ == "__main__":
    main()
