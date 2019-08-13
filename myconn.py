import pymysql.cursors

if __name__ == "__main__":
    # Connect to the database
    connection = pymysql.connect(host='localhost',
                                user='root',
                                password='root',
                                db='luckpot',
                                charset='utf8mb4',
                                cursorclass=pymysql.cursors.DictCursor)

    try:
        with connection.cursor() as cursor:
            # Read a single record
            sql = "SELECT * FROM history"
            cursor.execute(sql)
            result = cursor.fetchall()
            print(result)
    finally:
        connection.close()