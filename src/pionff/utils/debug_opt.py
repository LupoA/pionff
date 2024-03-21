import time


def timeit(debug=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            if debug:
                print(
                    f"Debug ::: {func.__name__} ::: execution time ::: {end_time - start_time} seconds."
                )
            return result

        return wrapper

    return decorator
