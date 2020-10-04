def arg(name, kwargs):
    if name not in kwargs:
        return None
    return kwargs[name]
