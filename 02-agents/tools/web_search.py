"""Explicit placeholders. No web/news provider is connected."""


def search_web(query: str, num_results: int = 5) -> str:
    if not isinstance(query, str) or not query.strip() or type(num_results) is not int or not 1 <= num_results <= 10:
        return "Ошибка: укажите запрос и num_results от 1 до 10"
    return "ДЕМО: веб-поиск не подключён. Реальные результаты и актуальные факты недоступны. Запрос: " + query[:500]


def search_news(query: str, days: int = 7) -> str:
    if not isinstance(query, str) or not query.strip() or type(days) is not int or not 1 <= days <= 365:
        return "Ошибка: укажите запрос и days от 1 до 365"
    return "ДЕМО: News API не подключён. Реальные новости не получены. Запрос: " + query[:500]
