user = 'mirzakhani_1934'
password = 'Bdragon4$'
host = 'data.codeup.com'

def get_db_url(database):
    return f'mysql+pymysql://{user}:{password}@{host}/{database}'