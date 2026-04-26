__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2020, Giacomo Bergami"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Giacomo Bergami"
__email__ = "bergamigiacomo@gmail.com"
__status__ = "Production"

from collections import OrderedDict, defaultdict
from contextlib import contextmanager

import psycopg2
from psycopg2 import Error, sql

from LaSSI.files.ReadFileContent import ReadFileContent


class FuzzyStringMatchDatabase:
    _instance = None

    def create(self, tablename, file, columns='(id integer NOT NULL, idx text, t text)', force=False):
        exists = False
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = %s)", (tablename,))
            exists = cursor.fetchone()[0]
        if not exists or force:
            print(f"Creating table {tablename}")
            with self.connection.cursor() as cursor:
                cursor.execute(f"DROP TABLE IF EXISTS {tablename}")
                cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
                cursor.execute(f"CREATE TABLE {tablename} {columns}")

                print(f"Table '{tablename}' is empty. Loading data from '{file}'")
                with ReadFileContent(file) as f:
                    next(f)  # Skip the header row.
                    cursor.copy_from(f, tablename, sep='\t')

                print(f"Creating index on table '{tablename}' if it doesn't exist")
                cursor.execute(
                    f"CREATE INDEX IF NOT EXISTS {tablename}_idx ON {tablename} USING GIST (t gist_trgm_ops);")
            self.connection.commit()
            print(f"Table {tablename} created!")
        else:
            print(f"Table {tablename} already loaded!")

    def init(self, database_name, user="lassi", password="drowssap", host="localhost", port="5432"):
        self.db_params = {
            'database': database_name,
            'user': user,
            'password': password,
            'host': host,
            'port': port
        }
        self.connection = psycopg2.connect(**self.db_params)

    def similarity(self, table, query, score=1.0):
        query = query.replace("'", "''")
        poll = defaultdict(set)
        with self.connection.cursor() as cursor:
            sql_query = f"""SELECT idx, similarity(t, '{query}') AS sml
                               FROM {table}
                               WHERE t % '{query}' AND similarity(t, '{query}')>={score}
                               ORDER BY sml DESC, t"""
            cursor.execute(sql_query)

            for monad, score in cursor:
                poll[score].add(monad)
        return poll

    def typed_similarity(self, table, query, score=1.0):
        query = query.replace("'", "''")
        poll = defaultdict(set)
        with self.connection.cursor() as cursor:
            sql_query = f"""SELECT idx, similarity(t, '{query}') AS sml, type
                               FROM {table}
                               WHERE t % '{query}' AND similarity(t, '{query}')>={score}
                               ORDER BY sml DESC, t"""
            cursor.execute(sql_query)

            for monad, score, r_type in cursor:
                poll[score].add((monad, r_type))
        return poll

    def morphosyntax(self, table, word, ending):
        poll = OrderedDict()
        with self.connection.cursor() as cursor:
            cursor.execute(f"""SELECT idx, t, similarity(t, '{word}') AS sml
                               FROM {table}
                               WHERE t % '{word}' AND t like '%{ending}'
                               ORDER BY sml DESC, t""")
            records = cursor.fetchall()
            for row in records:
                score = float(row[2])
                if score not in poll:
                    poll[score] = set()
                if row[1].endswith(ending):
                    poll[score].add(tuple([row[0], row[1]]))
            cursor.close()
        return poll

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'connection' in state:
            del state['connection']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.connection = psycopg2.connect(**self.db_params)

    @classmethod
    def instance(cls):
        if cls._instance is None:
            print('Creating new instance')
            cls._instance = cls.__new__(cls)
            # Put any initialization here.
        return cls._instance

    def __del__(self):
        self.connection.commit()
        self.connection.close()


class DBFuzzyStringMatching:
    def __init__(self, db, tablename):
        self.db = db
        self.tablename = tablename

    def fuzzyMatch(self, threshold: float, objectString: str):
        return self.db.similarity(self.tablename, objectString, score=threshold)

    def typedFuzzyMatch(self, threshold: float, objectString: str):
        return self.db.typed_similarity(self.tablename, objectString, score=threshold)


if __name__ == "__main__":
    with ReadFileContent("https://osf.io/download/a6yn8/") as r:
        next(r)
        for x in r:
            print(x)

    # c = FuzzyStringMatchDatabase.instance()
    # c.init("conceptnet")
    # print(c.morphosyntax("conceptnet", "traffic", "ed"))
    # c.create("conceptnet", "/home/giacomo/projects/similarity-pipeline/submodules/news-crawler/mini.h5_sql_input.txt")
    # c.create("geonames", "/home/giacomo/projects/similarity-pipeline/submodules/stanfordnlp_dg_server/allCountries.txt_sql_input.txt")
    # print(c.similarity("conceptnet", "giacomo", 20, .8))
    # print(c.similarity("geonames", "newcastle", 20, .8))
    # print(c.similarity("geonames", "newcastle city", 20, .8))
    # print(c.similarity("geonames", "newcastle upon tyne", 20, .8))
