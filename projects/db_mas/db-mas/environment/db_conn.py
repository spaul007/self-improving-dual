"""Postgres connection helper shared by the query tool and anomaly injection."""
from typing import Optional

import psycopg2

import config

DB_CONFIG = config.DB_CONFIG


def get_conn(application_name: Optional[str] = None):
    kwargs = dict(DB_CONFIG)
    if application_name:
        kwargs["application_name"] = application_name
    return psycopg2.connect(**kwargs)
