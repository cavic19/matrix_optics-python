from typing import Callable

def count_matching(iterable, func: Callable) -> int:
    """Counts occurence that satisfied the condition specified in func and returns it"""
    return sum(map(lambda x: 1 if func(x) else 0, iterable))


if __name__ == "__main__":
    d = {"a": 1, "b": 2, "c": 3}
    print(next(d))