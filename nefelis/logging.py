import logging


def setup(level):
    logging.basicConfig(
        level=level,
        force=True,
        style="{",
        format="{relativeCreatedSecs: >9.3f}s {levelname[0]} {name:<6s} {message}",
    )

    def add_relative_seconds(record):
        record.relativeCreatedSecs = record.relativeCreated / 1000.0
        return True

    logging.getLogger().handlers[0].addFilter(add_relative_seconds)
