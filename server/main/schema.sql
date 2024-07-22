DROP TABLE IF EXISTS poi;
DROP TABLE IF EXISTS rp;
DROP TABLE IF EXISTS ap;
DROP TABLE IF EXISTS fp;
DROP TABLE IF EXISTS route_point;
DROP TABLE IF EXISTS route_path;

CREATE TABLE poi(
    poi_id INTEGER PRIMARY KEY,
    poi_name TEXT,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    UNIQUE ( latitude, longitude )
);

CREATE TABLE ap(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mac TEXT UNIQUE NOT NULL,
    title TEXT
);

CREATE TABLE rp(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    UNIQUE ( latitude, longitude )
);

CREATE TABLE fp(
    rp_id INTEGER,
    ap_id INTEGER,
    rss INTEGER NOT NULL,
    FOREIGN KEY ( rp_id ) REFERENCES rp ( id ),
    FOREIGN KEY ( ap_id ) REFERENCES ap ( id ),
    PRIMARY KEY ( rp_id, ap_id )
);

CREATE TABLE route_point(
    id INTEGER PRIMARY KEY,
    latitude REAL NOT NULL,
    longitude REAL NOT NULl.
    UNIQUE ( latitude, longitude )
);

CREATE TABLE route_path(
    s_id INTEGER,
    n_id INTEGER, 
    FOREIGN KEY ( s_id ) REFERENCES route_point ( id ),
    FOREIGN KEY ( n_id ) REFERENCES route_point ( id ),
    PRIMARY KEY ( s_id, n_id )
);

