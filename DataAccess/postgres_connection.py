import psycopg2
from typing import List, Optional
from dataclasses import dataclass
from DataAccess.i_data_connection import IDataConnection


@dataclass
class Config:
    host: str
    port: int
    database: str
    user: str
    password: str


class PostgresConnection(IDataConnection):
    """
    A low-level connection class managing the physical database connection.
    It uses the default psycopg2 cursor to return results as a list of lightweight tuples,
    optimizing for speed and low memory footprint.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def execute_query(self, query: str, params: Optional[tuple] = None, fetch_results: bool = True) -> Optional[List[tuple]]:
        """
        Establishes a connection, executes a SQL query, and optionally fetches results.

        Uses the default cursor to return results as a list of lightweight tuples.

        Args:
            query: The SQL query string to execute.
            params: A tuple of parameters for the query (for security against SQL injection).
            fetch_results: If True, fetches and returns the results (for SELECT queries). 
                        If False, only executes the command (for INSERT, UPDATE, DELETE).

        Returns:
            A list of tuples if fetch_results is True, otherwise None.
        """
        conn = None
        results = None

        try:
            # Connect to the PostgreSQL database using configuration
            conn = psycopg2.connect(**self.cfg.__dict__)

            # Use the default cursor (which returns tuples) for performance
            with conn.cursor() as cur:

                # Execute the parameterized query
                cur.execute(query, params)

                if fetch_results:
                    # Fetch all rows as tuples
                    results = cur.fetchall()

                # Commit changes (required for INSERT, UPDATE, DELETE)
                conn.commit()

        except (Exception, psycopg2.Error) as error:
            print(f"Error while interacting with PostgreSQL: {error}")
            if conn:
                # Rollback any pending changes in case of an error
                conn.rollback()

        finally:
            # Close the connection
            if conn:
                conn.close()

        return results
