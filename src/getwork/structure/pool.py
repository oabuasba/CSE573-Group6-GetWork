import queue
import threading
from collections.abc import Callable, Iterator, Sequence
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")

POLL_SECONDS = 0.05


class _Dispatcher:
    """Hands out item indexes in order; shared state lives under one lock."""

    def __init__(self, count: int, stop: threading.Event, abort: threading.Event) -> None:
        self._count = count
        self._stop = stop
        self._abort = abort
        self._lock = threading.Lock()
        self._next_index = 0
        self._started = 0

    def claim(self) -> int | None:
        with self._lock:
            if self._stop.is_set() or self._abort.is_set() or self._next_index >= self._count:
                return None
            index = self._next_index
            self._next_index += 1
            self._started += 1
            return index

    def is_finished(self, yielded: int) -> bool:
        with self._lock:
            nothing_more = (
                self._next_index >= self._count or self._stop.is_set() or self._abort.is_set()
            )
            return yielded == self._started and nothing_more


def _work(
    items: Sequence[T],
    fn: Callable[[T], R],
    dispatcher: _Dispatcher,
    results: queue.Queue[tuple[T, R | BaseException]],
) -> None:
    while (index := dispatcher.claim()) is not None:
        item = items[index]
        try:
            outcome: R | BaseException = fn(item)
        except BaseException as exc:  # noqa: BLE001 - delivered to the consumer as the result
            outcome = exc
        results.put((item, outcome))


def run_pool(
    items: Sequence[T],
    fn: Callable[[T], R],
    workers: int,
    stop: threading.Event,
    abort: threading.Event,
) -> Iterator[tuple[T, R | BaseException]]:
    """Run fn over items on daemon threads, yielding (item, result) in arrival order.

    An exception raised by fn is yielded as the result. The consumer owns abort: after setting
    it and abandoning the generator, in-flight results are discarded.
    """
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    if not items:
        return
    dispatcher = _Dispatcher(count=len(items), stop=stop, abort=abort)
    results: queue.Queue[tuple[T, R | BaseException]] = queue.Queue()
    for number in range(min(workers, len(items))):
        threading.Thread(
            target=_work,
            args=(items, fn, dispatcher, results),
            name=f"structure-worker-{number}",
            daemon=True,
        ).start()
    yielded = 0
    while not dispatcher.is_finished(yielded):
        try:
            entry = results.get(timeout=POLL_SECONDS)
        except queue.Empty:
            continue
        yielded += 1
        yield entry
