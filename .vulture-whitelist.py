# Names a framework reads by contract, which vulture cannot see a caller for.
# pydantic passes cls to a @field_validator/@model_validator classmethod: pydantic/functional_validators.py.
cls
# langchain's callback protocol passes **kwargs to every on_* handler: BaseCallbackHandler.
kwargs
