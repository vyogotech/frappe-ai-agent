# ruff: noqa: F821, B018 -- a list of names for vulture, not code that runs.
# Names a framework reads by contract, which vulture cannot see a caller for.
# pydantic passes cls to a @field_validator/@model_validator classmethod.
cls
# langchain's callback protocol passes **kwargs to every on_* handler.
kwargs
