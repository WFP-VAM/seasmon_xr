from functools import wraps


def lazycompile(internal_decorator):
    """Delay wrapper."""

    def _lazycompile(f):
        inner_decorated = None

        @wraps(f)
        def wrapper(*args, **kwds):
            nonlocal inner_decorated
            if inner_decorated is None:
                inner_decorated = internal_decorator(f)
            return inner_decorated(*args, **kwds)

        return wrapper

    return _lazycompile
