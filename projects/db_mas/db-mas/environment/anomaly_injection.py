"""Ported anomaly injection logic (from MARBLE's anomaly_trigger/anomaly.py).

SQL logic and concurrency approach (multiprocessing.Pool for lock-contention-style
workers, ThreadPool for concurrent single-statement floods) are kept as in the
original for fidelity. Prometheus-based `restart_decision()` is dropped entirely
(it only ever fed an unreachable `sudo docker compose restart` path once stubbed).
"""
import random
import time
from multiprocessing.pool import Pool, ThreadPool
from typing import Any, Dict, List

import psycopg2

import config
from environment.db_conn import get_conn

ANOMALY_APPLICATION_NAME = "anomaly"
RESTART_APPLICATION_NAME = "restart"


def execute_sqls(sql: str, application_name: str = ANOMALY_APPLICATION_NAME) -> None:
    conn = get_conn(application_name=application_name)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
        cur.close()
    finally:
        conn.close()


def execute_sql_duration(duration: int, sql: str, max_id: int = 0, commit_interval: int = 500) -> int:
    conn = get_conn(application_name=ANOMALY_APPLICATION_NAME)
    try:
        cur = conn.cursor()
        start = time.time()
        cnt = 0
        while (time.time() - start) < duration:
            if max_id > 0:
                row_id = random.randint(1, max_id - 1)
                cur.execute(sql + str(row_id) + ";")
            else:
                cur.execute(sql)
            cnt += 1
            if cnt % commit_interval == 0:
                conn.commit()
        conn.commit()
        cur.close()
        return cnt
    finally:
        conn.close()


def concurrent_execute_sql(threads: int, duration: int, sql: str, max_id: int = 0, commit_interval: int = 500) -> None:
    pool = ThreadPool(threads)
    for _ in range(threads):
        pool.apply_async(execute_sql_duration, (duration, sql, max_id, commit_interval))
    pool.close()
    pool.join()


def build_index(table_name: str, idx_num: int) -> None:
    conn = get_conn(application_name=ANOMALY_APPLICATION_NAME)
    try:
        cur = conn.cursor()
        for i in range(idx_num):
            cur.execute(
                f"CREATE INDEX index_{table_name}_{i} ON {table_name}(name{i});"
            )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def drop_index(table_name: str) -> None:
    conn = get_conn(application_name=ANOMALY_APPLICATION_NAME)
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT indexname FROM pg_indexes WHERE tablename = '{table_name}';")
        idxs = cur.fetchall()
        for (idx_name,) in idxs:
            cur.execute(f"DROP INDEX {idx_name};")
        conn.commit()
        cur.close()
    finally:
        conn.close()


def create_table(table_name: str, colsize: int, ncolumns: int) -> None:
    column_definitions = ", ".join(f"name{i} varchar({colsize})" for i in range(ncolumns))
    execute_sqls(f"CREATE TABLE {table_name} (id int, {column_definitions}, time timestamp);")


def delete_table(table_name: str) -> None:
    execute_sqls(f"DROP TABLE IF EXISTS {table_name};")


def restart() -> None:
    """Terminate any backend left open under the 'anomaly' application_name.

    Uses a distinct application_name for this connection itself (RESTART_APPLICATION_NAME) --
    otherwise the pg_terminate_backend query would match and kill its own backend
    mid-query, since it would also be tagged application_name='anomaly'.
    """
    execute_sqls(
        "SELECT pg_terminate_backend(pg_stat_activity.pid) FROM pg_stat_activity "
        f"WHERE pg_stat_activity.application_name = '{ANOMALY_APPLICATION_NAME}';",
        application_name=RESTART_APPLICATION_NAME,
    )


def _insert_definitions(ncolumns: int, colsize: int) -> str:
    return ", ".join(
        f"(SELECT substr(md5(random()::text), 1, {colsize}))" for _ in range(ncolumns)
    )


def lock(table_name: str, ncolumns: int, colsize: int, duration: int, nrows: int) -> None:
    """Worker for the lock-contention Pool: hammer row-level UPDATEs for `duration` seconds."""
    start = time.time()
    while time.time() - start < duration:
        conn = get_conn(application_name=ANOMALY_APPLICATION_NAME)
        cur = conn.cursor()
        while time.time() - start < duration:
            col = random.randint(0, ncolumns - 1)
            row = random.randint(1, nrows - 1)
            cur.execute(
                f"UPDATE {table_name} SET name{col}=(SELECT substr(md5(random()::text), 1, {colsize})) "
                f"WHERE id = {row};"
            )
            conn.commit()
        conn.commit()
        conn.close()


def insert_large_data(threads: int, duration: int, ncolumns: int, nrows: int, colsize: int, table_name: str = "table1") -> None:
    delete_table(table_name)
    create_table(table_name, colsize, ncolumns)
    insert_data = (
        f"INSERT INTO {table_name} SELECT generate_series(1,{nrows}),"
        f"{_insert_definitions(ncolumns, colsize)}, NOW();"
    )
    time.sleep(10)
    concurrent_execute_sql(threads, duration, insert_data, commit_interval=1)
    time.sleep(10)
    restart()
    delete_table(table_name)


def lock_contention(threads: int, duration: int, ncolumns: int, nrows: int, colsize: int, table_name: str = "table1") -> None:
    delete_table(table_name)
    create_table(table_name, colsize, ncolumns)
    insert_data = (
        f"INSERT INTO {table_name} SELECT generate_series(1,{nrows}),"
        f"{_insert_definitions(ncolumns, colsize)}, NOW();"
    )
    execute_sqls(insert_data)

    pool = Pool(threads)
    time.sleep(10)
    for _ in range(threads):
        pool.apply_async(lock, (table_name, ncolumns, colsize, duration, nrows))
    pool.close()
    pool.join()
    time.sleep(10)
    restart()
    delete_table(table_name)


def vacuum(threads: int, duration: int, ncolumns: int, nrows: int, colsize: int, table_name: str = "table1") -> None:
    delete_table(table_name)
    create_table(table_name, colsize, ncolumns)
    execute_sqls(f"ALTER TABLE {table_name} SET (autovacuum_enabled = false);")

    insert_data = (
        f"INSERT INTO {table_name} SELECT generate_series(1,{nrows}),"
        f"{_insert_definitions(ncolumns, colsize)}, NOW();"
    )
    execute_sqls(insert_data)
    time.sleep(10)

    delete_nrows = int(nrows * 0.9)
    execute_sqls(f"DELETE FROM {table_name} WHERE id < {delete_nrows};")
    time.sleep(10)

    conn = get_conn(application_name=ANOMALY_APPLICATION_NAME)
    cur = conn.cursor()
    isolation_level = conn.isolation_level
    conn.set_isolation_level(0)  # VACUUM cannot run inside a transaction block
    for _ in range(threads):
        cur.execute("VACUUM FULL;")
    conn.set_isolation_level(isolation_level)
    conn.commit()
    conn.close()
    time.sleep(10)

    restart()
    delete_table(table_name)


def redundent_index(threads: int, duration: int, ncolumns: int, nrows: int, colsize: int, nindex: int, table_name: str = "table1") -> None:
    delete_table(table_name)
    create_table(table_name, colsize, ncolumns)
    insert_data = (
        f"INSERT INTO {table_name} SELECT generate_series(1,{nrows}),"
        f"{_insert_definitions(ncolumns, colsize)}, NOW();"
    )
    execute_sqls(insert_data)

    n_actual_index = int((nindex * ncolumns) / 10)
    build_index(table_name, n_actual_index)
    execute_sqls(f"CREATE INDEX index_{table_name}_id ON {table_name}(id);")
    time.sleep(10)

    pool = Pool(threads)
    for _ in range(threads):
        pool.apply_async(lock, (table_name, ncolumns, colsize, duration, nrows))
    pool.close()
    pool.join()
    time.sleep(10)

    drop_index(table_name)
    restart()
    delete_table(table_name)


def fetch_large_data() -> None:
    execute_sqls(
        "CREATE TABLE IF NOT EXISTS orders (o_orderkey int, o_orderpriority varchar(15), o_orderdate date);"
    )
    execute_sqls(
        "CREATE TABLE IF NOT EXISTS lineitem (l_orderkey int, l_commitdate date, l_receiptdate date);"
    )

    orders_insert_query = """
        INSERT INTO orders
        SELECT generate_series(1, 10000),
               CASE WHEN random() > 0.5 THEN '1-URGENT' ELSE '5-LOW' END::varchar,
               (date '1996-03-01' + (random() * (date '1998-09-01' - date '1996-03-01'))::int)
        ON CONFLICT DO NOTHING;
    """
    concurrent_execute_sql(1, 3, orders_insert_query, commit_interval=1)

    anomaly_query = "SELECT * FROM orders LIMIT 100;"
    # Was 1000 -- far past Postgres's max_connections=200 (see
    # environment/docker-compose.yml), so most of the 1000 threads' psycopg2
    # connections failed immediately, and the resulting flood of near-
    # simultaneous exceptions inside the ThreadPool's worker threads tripped
    # a CPython multiprocessing.pool bug during abnormal teardown
    # ("'DummyProcess' object has no attribute 'terminate'") -- found live
    # when db_mas ran its first FETCH_LARGE_DATA case ever, under the
    # self-improving-dual framework's evaluator. 100 matches
    # insert_large_data's own thread count and stays safely under the
    # connection cap.
    concurrent_execute_sql(100, 3, anomaly_query, commit_interval=1)

    time.sleep(15)
    restart()


def _spec_get(spec: Dict[str, Any], key: str, default: Any) -> Any:
    return spec[key] if key in spec else default


ANOMALY_DISPATCH = {
    "INSERT_LARGE_DATA": lambda a: insert_large_data(
        a["threads"], config.ANOMALY_DURATION_S, a["ncolumn"], a["nrow"], a["colsize"],
        _spec_get(a, "table_name", config.DEFAULT_TABLE_NAME),
    ),
    "REDUNDANT_INDEX": lambda a: redundent_index(
        a["threads"], config.ANOMALY_DURATION_S, a["ncolumn"], a["nrow"], a["colsize"],
        _spec_get(a, "nindex", config.DEFAULT_NINDEX),
        _spec_get(a, "table_name", config.DEFAULT_TABLE_NAME),
    ),
    "LOCK_CONTENTION": lambda a: lock_contention(
        a["threads"], config.ANOMALY_DURATION_S, a["ncolumn"], a["nrow"], a["colsize"],
        _spec_get(a, "table_name", config.DEFAULT_TABLE_NAME),
    ),
    "VACUUM": lambda a: vacuum(
        a["threads"], config.ANOMALY_DURATION_S, a["ncolumn"], a["nrow"], a["colsize"],
        _spec_get(a, "table_name", config.DEFAULT_TABLE_NAME),
    ),
    "FETCH_LARGE_DATA": lambda a: fetch_large_data(),
}


def inject_anomalies(anomaly_specs: List[Dict[str, Any]]) -> None:
    for spec in anomaly_specs:
        anomaly_type = spec["anomaly"]
        if anomaly_type not in ANOMALY_DISPATCH:
            raise ValueError(f"Unsupported anomaly type: {anomaly_type}")
        print(f"[anomaly_injection] injecting {anomaly_type} with spec {spec}")
        ANOMALY_DISPATCH[anomaly_type](spec)
