def flatten(x: list) -> list:
    return sum(map(flatten, x), []) if isinstance(x, list) else [x]
