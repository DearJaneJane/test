from skill.cache import TTLCache


def test_ttl_cache_returns_cached_value() -> None:
    cache: TTLCache[str, str] = TTLCache(maxsize=2, ttl_seconds=60)

    cache.set("q", "answer")

    assert cache.get("q") == "answer"


def test_ttl_cache_evicts_oldest_item() -> None:
    cache: TTLCache[str, str] = TTLCache(maxsize=1, ttl_seconds=60)

    cache.set("one", "1")
    cache.set("two", "2")

    assert cache.get("one") is None
    assert cache.get("two") == "2"
