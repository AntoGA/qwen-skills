"""Bounded read-only SQL for agent tools; demo creation is a separate trusted operation."""
import sqlite3
from pathlib import Path


def query_database(query: str, db_path: str = ":memory:") -> str:
    """Read at most 11 rows, display at most 10; never create/modify a file DB."""
    conn = None
    try:
        if not isinstance(query, str) or not query.strip() or len(query) > 10000:
            raise ValueError("SQL must contain 1..10000 characters")
        if db_path == ":memory:":
            conn = sqlite3.connect(":memory:")
        else:
            conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
        conn.execute("PRAGMA query_only=ON")
        conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1000000)
        conn.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 10000)
        allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_RECURSIVE}
        def authorize(action, first, second, database, source):
            if action in allowed:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_FUNCTION and second not in ('load_extension', 'writefile', 'readfile'):
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY
        conn.set_authorizer(authorize)
        steps = 0
        def progress():
            nonlocal steps
            steps += 1
            return int(steps > 100)
        conn.set_progress_handler(progress, 1000)
        cursor = conn.execute(query)
        if cursor.description is None:
            raise ValueError("Only row-returning read queries are supported")
        rows = cursor.fetchmany(11)
        if not rows:
            return "Запрос выполнен, но результаты не найдены"
        result = "📊 Результаты запроса (до 10 строк):\n"
        result += ' | '.join(column[0] for column in cursor.description) + '\n'
        for row in rows[:10]:
            result += ' | '.join(str(cell)[:1000] for cell in row) + '\n'
        if len(rows) > 10:
            result += '... результат ограничен первыми 10 строками'
        return result[:8000]
    except (sqlite3.Error, ValueError, TypeError, OSError) as error:
        return f"❌ Ошибка базы данных: {error}"
    finally:
        if conn is not None:
            conn.close()


def create_sample_database(db_path: str = "sample.db") -> str:
    """Seed stable synthetic IDs; repeated calls do not duplicate demo records."""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        with conn:
            conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER, city TEXT)")
            conn.execute("CREATE TABLE IF NOT EXISTS orders (id INTEGER PRIMARY KEY, user_id INTEGER, product TEXT, amount REAL, FOREIGN KEY(user_id) REFERENCES users(id))")
            conn.executemany("INSERT OR IGNORE INTO users VALUES (?,?,?,?)", [
                (1, 'Алексей', 28, 'Москва'), (2, 'Мария', 34, 'Санкт-Петербург'),
                (3, 'Дмитрий', 42, 'Новосибирск'), (4, 'Елена', 25, 'Москва'),
                (5, 'Сергей', 31, 'Екатеринбург')])
            conn.executemany("INSERT OR IGNORE INTO orders VALUES (?,?,?,?)", [
                (1, 1, 'Ноутбук', 85000), (2, 1, 'Мышь', 2500),
                (3, 2, 'Телефон', 65000), (4, 3, 'Планшет', 45000),
                (5, 4, 'Наушники', 12000)])
        return "✅ Демо-база данных создана. Таблицы: users, orders"
    except (sqlite3.Error, OSError) as error:
        return f"❌ Ошибка создания БД: {error}"
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    print(create_sample_database("sample.db"))
    print(query_database("SELECT * FROM users", "sample.db"))
