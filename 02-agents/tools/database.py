"""
Database Tool
==============
Инструмент для работы с базами данных
"""

import sqlite3
from typing import List, Dict, Any, Optional


def query_database(query: str, db_path: str = ":memory:") -> str:
    """
    Выполнение SQL запроса к базе данных
    
    Args:
        query: SQL запрос
        db_path: Путь к файлу базы данных (по умолчанию: in-memory)
    
    Returns:
        str: Результаты запроса
    """
    
    try:
        # Подключение к БД
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Выполнение запроса
        cursor.execute(query)
        
        # Получение результатов
        if query.strip().upper().startswith("SELECT"):
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            if not rows:
                return "Запрос выполнен, но результаты не найдены"
            
            # Форматирование результатов
            result = f"📊 Результаты запроса ({len(rows)} строк):\n\n"
            result += " | ".join(columns) + "\n"
            result += "-" * 60 + "\n"
            
            for row in rows[:10]:  # Ограничение до 10 строк
                result += " | ".join(str(cell) for cell in row) + "\n"
            
            if len(rows) > 10:
                result += f"\n... и еще {len(rows) - 10} строк"
            
            return result
        else:
            conn.commit()
            return f"✅ Запрос выполнен успешно. Затронуто строк: {cursor.rowcount}"
            
    except sqlite3.Error as e:
        return f"❌ Ошибка базы данных: {str(e)}"
    except Exception as e:
        return f"❌ Ошибка: {str(e)}"
    finally:
        if 'conn' in locals():
            conn.close()


def create_sample_database(db_path: str = ":memory:") -> str:
    """
    Создание демонстрационной базы данных
    
    Args:
        db_path: Путь к файлу БД
    
    Returns:
        str: Сообщение о создании
    """
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Создание таблиц
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                age INTEGER,
                city TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                product TEXT,
                amount REAL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        # Вставка тестовых данных
        cursor.executemany("""
            INSERT INTO users (name, age, city) VALUES (?, ?, ?)
        """, [
            ("Алексей", 28, "Москва"),
            ("Мария", 34, "Санкт-Петербург"),
            ("Дмитрий", 42, "Новосибирск"),
            ("Елена", 25, "Москва"),
            ("Сергей", 31, "Екатеринбург")
        ])
        
        cursor.executemany("""
            INSERT INTO orders (user_id, product, amount) VALUES (?, ?, ?)
        """, [
            (1, "Ноутбук", 85000.00),
            (1, "Мышь", 2500.00),
            (2, "Телефон", 65000.00),
            (3, "Планшет", 45000.00),
            (4, "Наушники", 12000.00)
        ])
        
        conn.commit()
        conn.close()
        
        return "✅ Демо-база данных создана успешно!\nТаблицы: users, orders"
        
    except Exception as e:
        return f"❌ Ошибка создания БД: {str(e)}"


if __name__ == "__main__":
    # Тестирование
    print("Создание демо-базы данных...")
    print(create_sample_database("test.db"))
    print("\n" + "="*60 + "\n")
    
    print("Выполнение запросов:")
    print(query_database("SELECT * FROM users", "test.db"))
    print("\n" + "="*60 + "\n")
    
    print("Агрегатный запрос:")
    print(query_database("""
        SELECT u.name, COUNT(o.id) as order_count, SUM(o.amount) as total
        FROM users u
        LEFT JOIN orders o ON u.id = o.user_id
        GROUP BY u.id
    """, "test.db"))
