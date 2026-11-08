import threading

import pytest

from getwork.structure import pool

pytestmark = pytest.mark.usefixtures("offline")

TIMEOUT = 5.0
HOLD_SECONDS = 0.1


def events() -> tuple[threading.Event, threading.Event]:
    return threading.Event(), threading.Event()


def test_pool_starts_items_in_order_and_yields_all() -> None:
    started: list[int] = []
    lock = threading.Lock()

    def fn(item: int) -> int:
        with lock:
            started.append(item)
        return item * 10

    stop, abort = events()
    single = list(pool.run_pool(items=[1, 2, 3, 4], fn=fn, workers=1, stop=stop, abort=abort))
    assert single == [(1, 10), (2, 20), (3, 30), (4, 40)]
    assert started == [1, 2, 3, 4]

    many = list(pool.run_pool(items=list(range(7)), fn=fn, workers=3, stop=stop, abort=abort))
    assert sorted(many) == [(i, i * 10) for i in range(7)]


def test_pool_delivers_exceptions_as_results() -> None:
    def fn(item: int) -> int:
        if item == 2:
            raise ValueError("boom")
        return item

    stop, abort = events()
    results = dict(pool.run_pool(items=[1, 2, 3], fn=fn, workers=2, stop=stop, abort=abort))
    assert isinstance(results[2], ValueError)
    assert results[1] == 1
    assert results[3] == 3


def test_pool_concurrency_is_bounded_by_workers() -> None:
    lock = threading.Lock()
    current = 0
    peak = 0
    never = threading.Event()

    def fn(item: int) -> int:
        nonlocal current, peak
        with lock:
            current += 1
            peak = max(peak, current)
        never.wait(timeout=HOLD_SECONDS)
        with lock:
            current -= 1
        return item

    stop, abort = events()
    results = list(pool.run_pool(items=list(range(6)), fn=fn, workers=2, stop=stop, abort=abort))
    assert len(results) == 6
    assert peak == 2


def test_pool_stop_drains_in_flight_and_starts_nothing_new() -> None:
    stop, abort = events()
    started: list[int] = []

    def fn(item: int) -> int:
        started.append(item)
        if item == 2:
            stop.set()
        return item

    results = list(pool.run_pool(items=[1, 2, 3, 4, 5], fn=fn, workers=1, stop=stop, abort=abort))
    assert [item for item, _ in results] == [1, 2]
    assert started == [1, 2]


def test_pool_abort_discards_in_flight_and_starts_nothing_new() -> None:
    stop, abort = events()
    release = threading.Event()
    started: list[int] = []
    lock = threading.Lock()

    def fn(item: int) -> int:
        with lock:
            started.append(item)
        if item != 1:
            release.wait(timeout=TIMEOUT)
        return item

    generator = pool.run_pool(items=list(range(1, 7)), fn=fn, workers=2, stop=stop, abort=abort)
    assert next(generator) == (1, 1)
    abort.set()
    generator.close()
    release.set()
    for thread in threading.enumerate():
        if thread.name.startswith("structure-worker-"):
            thread.join(timeout=TIMEOUT)
    # Every item but 1 blocks until abort is set, so only 1, 2 and one item claimed before abort run.
    assert set(started) <= {1, 2, 3}
    assert list(generator) == []


def test_pool_rejects_non_positive_workers() -> None:
    stop, abort = events()
    with pytest.raises(ValueError):
        list(pool.run_pool(items=[1], fn=lambda item: item, workers=0, stop=stop, abort=abort))


def test_pool_threads_are_daemon() -> None:
    flags: list[bool] = []

    def fn(item: int) -> int:
        flags.append(threading.current_thread().daemon)
        return item

    stop, abort = events()
    list(pool.run_pool(items=[1, 2], fn=fn, workers=2, stop=stop, abort=abort))
    assert flags == [True, True]
