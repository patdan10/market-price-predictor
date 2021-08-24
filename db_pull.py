import psycopg2
import pandas as pd

# Columns of Data to Export
cols = ['PriceDate', 'Hour',
        'LoadCISO', 'LoadPGE', 'LoadSCE',
        'WindPGE', 'WindSCE', 'SolarPGE', 'SolarSCE',
        'PricePGE', 'PriceSCE',
        'CongestPGE', 'CongestSCE',
        'LossPGE', 'LossSCE']

# Function to Get Data
def get_data():
    # Connect to Database
    conn = psycopg2.connect(dbname='ISO', user='pdanielson', password='davidson456', host='fortdash.xyz')
    cur = conn.cursor()
    cur.execute("SET search_path TO ciso;")

    # Execute insanely long SQL command
    cur.execute("""SELECT DISTINCT load.pricedate, load.hour, load.mw, l1.mw, l2.mw,
    ROUND(SUM(CASE WHEN (r2.mw IS NOT NULL AND r2.hub = 'NP15') THEN r2.mw ELSE 0.0 END), 2),
    ROUND(SUM(CASE WHEN (r2.mw IS NOT NULL AND (r2.hub = 'ZP26' OR r2.hub = 'SP15')) THEN r2.mw ELSE 0.0 END), 2),
    ROUND(SUM(CASE WHEN r1.hub = 'NP15' THEN r1.mw ELSE 0.0 END), 2),
    ROUND(SUM(CASE WHEN (r1.hub = 'ZP26' OR r1.hub = 'SP15') THEN r1.mw ELSE 0.0 END), 2),
    p1.dalmp, p2.dalmp, p1.damcc, p2.damcc, p1.damcl, p2.damcl

    FROM load
    LEFT OUTER JOIN renewables r1 ON ((load.pricedate=r1.pricedate) AND (load.hour=r1.hour)
        AND (r1.type = 'Solar') AND r1.market = 'DAM')
    LEFT OUTER JOIN renewables r2 ON ((load.pricedate=r2.pricedate) AND (load.hour=r2.hour)
        AND (r2.type = 'Wind') AND (r2.hub = r1.hub) AND r2.market = 'DAM')
    LEFT OUTER JOIN load l1 ON ((load.pricedate=l1.pricedate) AND (load.hour=l1.hour)
        AND l1.market='DAM' AND (l1.area='PGE-TAC'))
    LEFT OUTER JOIN load l2 ON ((load.pricedate=l2.pricedate) AND (load.hour=l2.hour)
        AND l2.market='DAM' AND (l2.area='SCE-TAC'))
    LEFT OUTER JOIN prices p1 ON ((load.pricedate=p1.pricedate) AND (load.hour=p1.hour)
        AND (p1.node_id=5170))
    LEFT OUTER JOIN prices p2 ON ((load.pricedate=p2.pricedate) AND (load.hour=p2.hour)
        AND (p2.node_id=5754))
    
    WHERE (load.market='DAM'
        AND (load.pricedate >= '2020-04-30 00:00:00')
        AND NOT (load.pricedate BETWEEN '2021-02-15 00:00:00' AND '2021-02-18 23:00:00')
        AND load.market = 'DAM'
        And (load.area='CA ISO-TAC'))
    
    GROUP BY load.pricedate, load.hour, load.mw, l1.mw, l2.mw,
        p1.dalmp, p2.dalmp, p1.damcc, p2.damcc, p1.damcl, p2.damcl
    
    ORDER BY load.pricedate;""")


    # Fetch, format, and return
    out = cur.fetchall()
    df = pd.DataFrame(data=out)
    df.columns = cols

    return df









def get_schema():
    conn = psycopg2.connect(dbname='ISO', user='pdanielson', password='davidson456', host='fortdash.xyz')
    cur = conn.cursor()
    cur.execute("""SELECT schemaname, tablename
    FROM pg_catalog.pg_tables
    WHERE schemaname != 'pg_catalog'
    AND schemaname != 'information_schema';""")
    out = cur.fetchall()

    # print()
    # cur.execute("SET search_path TO ciso;")


if __name__ == "__main__":
    get_schema()





# PRINT SCHEMA AND TABLES
"""cur.execute("SELECT schemaname, tablename FROM pg_catalog.pg_tables WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema';") out = cur.fetchall() print(out) print() """
# PRINT COLUMN NAMES
"""cur.execute("select column_name,data_type from information_schema.columns where table_name = 'load';")
out = cur.fetchall()
print(out)"""

"""
    #  OR area='PGE-TAC' OR area='SCE-TAC'

    # cur.execute("select column_name,data_type from information_schema.columns where table_name = 'renewables';")
    # print(out)
    cur.execute(
        "select * from renewables where pricedate >= '2021-01-01 00:00:00' and market = 'DAM' order by pricedate limit 15;")
    out2 = cur.fetchall()
    conn.close()

    for o in out2:
        print(o)
        """

"""
cur.execute("select * from load where ((load.pricedate='2021-01-01 00:00:00') AND (load.hour=1) AND load.market='DAM' AND (load.area='CA ISO-TAC')) limit 30;")
o = cur.fetchall()
print(o)

cur.execute("select * from load where ((load.pricedate='2021-01-01 00:00:00') AND (load.hour=1) AND load.market='DAM' AND (load.area='SCE-TAC')) limit 30;")
o = cur.fetchall()
print(o)

cur.execute("select * from load where ((load.pricedate='2021-01-01 00:00:00') AND (load.hour=1) AND load.market='DAM' AND (load.area='PGE-TAC')) limit 30;")
o = cur.fetchall()
print(o)
"""