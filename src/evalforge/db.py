from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row

from .config import DATABASE_URL


@contextmanager
def connect():
    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
        yield conn


@contextmanager
def cursor():
    with connect() as conn, conn.cursor() as cur:
        yield cur
        conn.commit()


def ping() -> bool:
    with cursor() as cur:
        cur.execute("SELECT 1 AS ok")
        row = cur.fetchone()
        return bool(row and row["ok"] == 1)
