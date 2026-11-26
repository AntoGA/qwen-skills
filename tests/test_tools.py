import sqlite3
import pytest
from tools.calculator import calculate
from tools.database import create_sample_database, query_database
from tools.web_search import search_web, search_news


@pytest.mark.parametrize('expression,value', [('(15 * 23) + 42',387), ('sqrt(144)',12),
    ('sin(pi/2)',1), ('pow(2, 10)',1024), ('sum([1,2,3])',6), ('min(4,2)',2)])
def test_arithmetic(expression, value):
    assert calculate(expression).endswith('= ' + str(value)) or float(calculate(expression).split(' = ')[-1]) == value


@pytest.mark.parametrize('expression', ["'x' * 100000000", '[1] * 100000000',
    '2 ** 100000000', 'pow(2,100000000)', '(1).__class__', '__import__("os")',
    '[x for x in (1,2)]', 'lambda: 1', 'True', '1e999', '1+1j', 'sqrt(-1)', '', '1'*513])
def test_calculator_rejects_non_arithmetic_and_excessive_work(expression):
    assert calculate(expression).startswith('Ошибка')


def test_division_by_zero():
    assert 'деление на ноль' in calculate('1/0')


@pytest.fixture
def database(tmp_path):
    path = tmp_path / 'demo.db'
    create_sample_database(str(path))
    return path


def test_demo_seed_is_repeatable(database):
    create_sample_database(str(database))
    with sqlite3.connect(database) as conn:
        assert conn.execute('SELECT COUNT(*) FROM users').fetchone()[0] == 5
        assert conn.execute('SELECT COUNT(*) FROM orders').fetchone()[0] == 5


@pytest.mark.parametrize('sql', ['DELETE FROM users', 'DROP TABLE users',
    'UPDATE users SET name="changed"', 'PRAGMA writable_schema=ON',
    "ATTACH DATABASE ':memory:' AS extra", "SELECT load_extension('x')"])
def test_database_is_read_only(database, sql):
    assert query_database(sql, str(database)).startswith('❌')
    with sqlite3.connect(database) as conn:
        assert conn.execute('SELECT COUNT(*) FROM users').fetchone()[0] == 5


def test_cte_and_no_silent_database_creation(database, tmp_path):
    result = query_database('WITH example AS (SELECT name FROM users) SELECT * FROM example', str(database))
    assert 'Алексей' in result
    missing = tmp_path / 'missing.db'
    assert query_database('SELECT 1', str(missing)).startswith('❌')
    assert not missing.exists()


def test_sql_limits(database):
    assert query_database('SELECT randomblob(2000000)', str(database)).startswith('❌')
    sql = 'WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n) SELECT sum(x) FROM n'
    assert query_database(sql, str(database)).startswith('❌')


def test_search_is_explicitly_not_live():
    assert search_web('weather').startswith('ДЕМО:')
    assert search_news('today').startswith('ДЕМО:')
