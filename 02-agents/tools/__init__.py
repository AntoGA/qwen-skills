"""
Tools Package
==============
Набор инструментов для AI агентов
"""

from .web_search import search_web
from .calculator import calculate
from .database import query_database

__all__ = ['search_web', 'calculate', 'query_database']
