import mysql.connector

conn = mysql.connector.connect(
    user='root', password='', host='localhost', database='python')


def insert(sql, val=""):
    try:
        cursor = conn.cursor()
        cursor.execute(sql, val)
        conn.commit()
    except:
        conn.rollback()
        conn.close()


def select(sql):
    cursor = conn.cursor(buffered=True)
    cursor.execute(sql)
    cursor.fetchall()
    conn.close()


def update(sql):
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()
        conn.close()
    except:
        conn.rollback()
        conn.close()
