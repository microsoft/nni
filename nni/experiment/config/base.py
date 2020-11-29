# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

class ConfigBase:
    """
    Base class of all config classes.

    Subclasses may override:

      - `_field_validation_rules`
      - `_class_validation_rules`
      - `from_json()`
      - `json()`
    """

    # Dict from field name to validation rule.
    # A validation rule is a callable, whose parameters are `(field_value, config_object)`,
    # and return value should either be `valid` (bool), or `(valid, error_message)`.
    # A rule is invoked only when `field_value` has correct type and is not None.
    # `error_message` will be attached to the exception raised when `valid` is False,
    # and it will be prepended with class name and field name.
    _field_validation_rules = {}  # don't add type hint so dataclass won't treat it as field

    # List of class-wise validation ruels.
    # Similar to `_field_validation_rules`, a rule is a callable, whose parameter is `config_object`,
    # and return value should either be `valid`, or `(valid, error_message)`.
    _class_validation_rules = []  # don't add type hint so dataclass won't treat it as field

    def __init__(self, **kwargs):
        """
        Initialize a config object and set some fields.

        Name of keyword arguments can either be snake_case or camelCase.
        They will be converted to snake_case automatically.

        If a field is missing and don't have default value, it will be set to `dataclasses.MISSING`.
        """
        kwargs = {util.case_insensitive(key): value for key, value in kwargs.items()}
        for field in dataclasses.fields(self):
            value = kwargs.pop(util.case_insensitive(field.name), field.default)
            setattr(self, field.name, value)
        if kwargs:
            cls = type(self).__name__
            fields = ', '.join(kwargs.keys())
            raise ValueError(f'{cls}: Unrecognized fields {fields}')

    @classmethod
    def load(cls, path: PathLike) -> cls:
        """
        Load a config object from YAML or JSON file.
        The file is assumed to be YAML unless `path` endswith ".json" (case-insensitive).
        """
        if str(path).lower().endswith('.json'):
            data = json.load(open(path))
        data = yaml.load(open(path))
        if not isinstance(data, dict):
            raise ValueError(f'Content of config file {path} is not a JSON object')
        return cls.from_json()(data)

    @classmethod
    def from_json(cls, json_object: Dict[str, Any]) -> cls:
        """
        Create config from JSON object.
        The keys of `json_object` can either be snake_case or camelCase.
        """
        return cls(**json_object)

    def json(self) -> Dict[str, Any]:
        """
        Convert config to JSON object.
        The keys of returned object will be camelCase.
        """
        return dataclasses.asdict(
            self,
            dict_factory = lambda items: dict((util.camel_case(k), v) for k, v in items)
        )

    def validate(self) -> None:
        """
        Validate the config object and raise Exception if it's ill-formed.
        """
        cls = type(self).__name__

        for field in dataclasses.fields(self):
            key, value = field.name, getattr(self, field.name)

            # check existence
            if value == dataclasses.MISSING:
                raise ValueError(f'{cls}: {key} is not set')

            # check type
            # TODO
            type_name = str(field.type).replace('typing.', '')
            if type_name.startswith('Optional[') and value is None:
                continue

            # check value
            rule = self._field_validation_rules.get(key)
            if rule is not None:
                try:
                    result = rule(value, self)
                except Exception as e:
                    msg = f'{cls}: {key} has bad value {repr(value)}'
                    raise ValueError(e)(msg, *e.args).with_traceback(e)

                if isinstance(result, bool):
                    if not result:
                        raise ValueError(f'{cls}: {key} ({repr(value)}) is out of range')
                else:
                    if not result[0]:
                        raise ValueError(f'{cls}: {key} {result[1]}')

            # check nested config
            if isinstance(value, ConfigBase):
                value.validate()

        for rule in _class_validation_rules:
            valid, msg = rule(self)
            if not valid:
                raise ValueError(msg)
