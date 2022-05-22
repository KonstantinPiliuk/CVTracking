import sqlalchemy as sa
import datetime
import numpy as np

def db_connect(dialect='mysql', host=None, user=None, pwd=None, database=None):
    '''
    Creates SQLAlchemy connection to sql database
    ------------
    parameters:
    -- dialect: SQLAlchemy dialect
    -- host: host and port xxx.xx.xx.xx:xxxx
    -- user: username
    -- pwd: password
    -- database: database name
    ----------
    output:
    -- SQLAlchemy engine
    '''
    if pwd is None:
        raise Exception('Please provide password with --p argument')
    if user is None:
        raise Exception('Please provide username with --u argument')
    if host is None:
        raise Exception('Please provide hostname in XXX.XXX.XX.XXX:XXXX format with --h argument')
    if database is None:
        raise Exception('Please provide database name with --db argument')
    return sa.create_engine(f"{dialect}://{user}:{pwd}@{host}/{database}")

def insert_data(df, table, increment=False, dialect='mysql', host=None, user=None, pwd=None, database=None):
    '''
    Inserts pandas.DataFrame into sql table
    ---------
    parameters:
    -- df: pandas.DataFrame to write in sql database
    -- table: table name for insertion
    -- increment: if True adds data into previous game_id. Data for new game_id is inserted by default.
    -- dialect: SQLAlchemy dialect
    -- host: host and port xxx.xx.xx.xx:xxxx
    -- user: username
    -- pwd: password
    -- database: database name
    '''
    db = db_connect(dialect=dialect, host=host, user=user, pwd=pwd, database=database)
    data = df.copy()

    #check if tables are initialized and equally filled
    if not np.all([sa.inspect(db).has_table(x) for x in [
        'games_ref', 'detected_objects', 'detected_keypoints', 'keypoints_ref',
        'homography_filters', 'fct_transform', 'matches_ref', 'fct_track']]):

        scheme_init(dialect=dialect, host=host, user=user, pwd=pwd, database=database)

    #last game_id
    last_game = db.execute('select max(game_id) from games_ref').fetchone()._mapping['max(game_id)']
    last_game = 0 if last_game is None else last_game
    #initialize game for game_ref table
    if table == 'games_ref' and not increment:
        last_game += 1
    #add constants - first and last fields
    data.insert(loc=data.shape[1], column='PROCESSED_DTTM', value = datetime.datetime.today())
    data.insert(loc=0, column='GAME_ID', value = last_game)

    #rename fields as in the database
    col_names = list(db.execute(f'select * from {table} where 1=0')._metadata.keys)
    data.columns = col_names

    #write
    data.to_sql(name=table, con=db, schema=database, if_exists='append', index=False)
    print(f'{data.shape[0]} rows are inserted to {table}') 

def scheme_init(dialect='mysql', host=None, user=None, pwd=None, database=None):
    '''
    Creates empty tables if they don't exist yet
    ------------
    parameters:
    -- dialect: SQLAlchemy dialect
    -- host: host and port xxx.xx.xx.xx:xxxx
    -- user: username
    -- pwd: password
    -- database: database name
    '''
    db = db_connect(dialect=dialect, host=host, user=user, pwd=pwd, database=database)

    #DROP scheme tables and create new ones
    db.execute("DROP TABLE IF EXISTS games_ref")
    db.execute("DROP TABLE IF EXISTS detected_objects")
    db.execute("DROP TABLE IF EXISTS detected_keypoints")
    db.execute("DROP TABLE IF EXISTS keypoints_ref")
    db.execute("DROP TABLE IF EXISTS homography_filters")
    db.execute("DROP TABLE IF EXISTS fct_transform")
    db.execute("DROP TABLE IF EXISTS matches_ref")
    db.execute("DROP TABLE IF EXISTS fct_track")

    #games_ref referecnce table with previous games
    db.execute('''
    CREATE TABLE games_ref (
        GAME_ID INT, 
        HOME_SIDE VARCHAR(50),
        AWAY_SIDE VARCHAR(50),
        GAME_DT DATE,
        EXECUTED_DTTM DATETIME(0))
    ''')

    #detected_objects table for detect() step logging
    db.execute('''
    CREATE TABLE detected_objects (
        GAME_ID INT, 
        FRAME_ID INT,
        DETECTION_ID INT,
        X_MIN FLOAT,
        Y_MIN FLOAT,
        X_MAX FLOAT,
        Y_MAX FLOAT,
        CONFIDENCE_SCORE FLOAT,
        CLASS_ID INT,
        NAME VARCHAR(20),
        EXECUTED_DTTM DATETIME(0))
    ''')

    #detected_objects table for detect() step logging
    db.execute('''
    CREATE TABLE detected_keypoints (
        GAME_ID INT, 
        FRAME_ID INT,
        KEYPOINT_ID INT,
        X_POS FLOAT,
        Y_POS FLOAT,
        EXECUTED_DTTM DATETIME(0))
    ''')

    #reference object for keypoints
    db.execute('''
    CREATE TABLE keypoints_ref (
        GAME_ID INT,
        KEYPOINT_ID INT, 
        KEYPOINT_NM VARCHAR(50),
        PITCH_X_POS FLOAT,
        PITCH_Y_POS FLOAT,
        EXECUTED_DTTM DATETIME(0))
    ''')

    #filters for frames exclusion
    db.execute('''
    CREATE TABLE homography_filters (
        GAME_ID INT,
        FRAME_ID INT, 
        BAD_HOMOGRAPHY_FLG CHAR,
        NO_KEYPOINTS_FLG CHAR,
        COMMON_FILTER CHAR,
        EXECUTED_DTTM DATETIME(0))
    ''')

    #final Detector output after transformation
    db.execute('''
    CREATE TABLE fct_transform (
        GAME_ID INT,
        FRAME_ID INT,
        DETECTION_ID INT,
        X_POS FLOAT,
        Y_POS FLOAT,
        VIDEO_MSECS FLOAT,
        COLOR_MEDIAN FLOAT,
        EXECUTED_DTTM DATETIME(0))
    ''')

    #history of objects matching
    db.execute('''
    CREATE TABLE matches_ref (
        GAME_ID INT,
        FRAME_ID INT,
        DETECTION_ID INT,
        OBJECT_ID INT,
        EXECUTED_DTTM DATETIME(0))
    ''')

    #final Tracker+Extractor output after tracking
    db.execute('''
    CREATE TABLE fct_track (
        GAME_ID INT,
        OBJECT_ID INT,
        FRAME_ID INT,
        X_POS FLOAT,
        Y_POS FLOAT,
        VIDEO_MSECS FLOAT,
        COLOR_MEDIAN FLOAT,
        X_VEL FLOAT,
        Y_VEL FLOAT,
        X_ACC FLOAT,
        Y_ACC FLOAT,
        X_EXP_POS FLOAT,
        Y_EXP_POS FLOAT,
        LAST_DETECTION_MSECS FLOAT,
        EXECUTED_DTTM DATETIME(0))
    ''')

    print('Tables are initialized')