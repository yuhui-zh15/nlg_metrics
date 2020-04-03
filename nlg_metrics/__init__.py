from nlg_metrics.scorer import RougeScorer, BertScorer, FactScorer, MoverScorer

import logging.config
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s %(levelname)s: %(message)s",
                'datefmt': '%Y-%m-%d %H:%M:%S'
                }
            },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
            }
        },
        "loggers": {
            "": {"handlers": ["console"]}
        },
    }
)
