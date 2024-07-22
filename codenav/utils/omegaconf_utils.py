import ast
import copy
import json
import os.path
import sys
from typing import Optional, Sequence

from omegaconf import OmegaConf, DictConfig, MissingMandatoryValue
from omegaconf.errors import InterpolationKeyError


def recursive_merge(base: DictConfig, other: DictConfig) -> DictConfig:
    base = copy.deepcopy(base)

    if all(key not in base for key in other):
        return OmegaConf.merge(base, other)

    for key in other:
        try:
            value = other[key]
        except InterpolationKeyError:
            value = other._get_node(key)._value()
        except MissingMandatoryValue:
            value = other._get_node(key)._value()

        if (
            key in base
            and isinstance(base[key], DictConfig)
            and isinstance(value, DictConfig)
        ):
            base[key] = recursive_merge(base[key], value)
        else:
            base[key] = value

    return base


def load_config_from_file(config_path: str, at_key: Optional[str] = None) -> DictConfig:
    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    base_dir = os.path.dirname(os.path.abspath(config_path))

    main_conf = OmegaConf.load(config_path)

    defaults_paths = main_conf.get("_defaults_", [])
    if "_self_" not in defaults_paths:
        defaults_paths.insert(0, "_self_")

    main_conf.pop("_defaults_", None)

    config = OmegaConf.create()
    for p in defaults_paths:
        if p == "_self_":
            config = recursive_merge(config, main_conf)
        else:
            key = None
            if "@" in p:
                p, key = p.split("@")

            if not os.path.isabs(p):
                p = os.path.join(base_dir, p)

            other_conf = load_config_from_file(p, at_key=key)

            config = recursive_merge(config, other_conf)

    if at_key is not None:
        top = OmegaConf.create()
        cur = top
        sub_keys = at_key.split(".")
        for sub_key in sub_keys[:-1]:
            cur[sub_key] = OmegaConf.create()
            cur = cur[sub_key]
        cur[sub_keys[-1]] = config
        config = top

    return config


def parse_omegaconf(
    base_config_path: Optional[str],
    args: Optional[Sequence[str]] = None,
) -> DictConfig:
    """
    Parses and merges configuration files and command line arguments using OmegaConf.

    This function loads a base configuration file (if provided), processes command line
    arguments to override configuration values or load additional files, and returns
    the final merged configuration.

    Basics:
        OmegaConf is a flexible and powerful configuration management library for Python.
        It supports hierarchical configurations, interpolation, and merging of configuration files.
        Key features of OmegaConf include:
        - Hierarchical Configuration: Allows configurations to be organized in nested structures.
        - Interpolation: Supports references to other values within the configuration.
        - Merging: Combines multiple configurations, with later configurations overriding earlier ones.

    YAML file overriding with `_defaults_`:
        You can use the `_defaults_` field in a YAML file to specify a list
        of other YAML files that should be loaded and merged. The `_defaults_` field allows
        you to control the order in which configurations are applied, with later files in
        the list overriding earlier ones.

        Example YAML files:

        ```yaml
        # logging.yaml
        log_level: info
        ```

        ```yaml
        # database.yaml
        database:
            host: localhost
            port: 5432
        ```

        ```yaml
        # base.yaml
        _defaults_:
          - logging.yaml@logging
          - database.yaml

        database:
            port: 3306
        ```

        Running `parse_omegaconf("base.yaml")` will result in the following configuration:
        ```yaml
        logging:
            log_level: info
        database:
            host: localhost
            port: 5432
        ```

        How overriding works:
        1. When loading the main configuration file, `base.yaml`, the configurations listed in `_defaults_` are loaded
            in the order specified.
        2. The main configuration file (the one containing `_defaults_`) IS LOADED FIRST BY DEFAULT.
        3. If there are conflicting keys, the values in the later files override the earlier ones.
        4. If a default yaml file is post-fixed with `@key`, the configuration will be placed at the specified key. Notice
            that logging.yaml@logging results in log_level: info being placed under the logging key in the final configuration.
        5. By default, the fields in the main configuration file ARE OVERWRITTEN by the fields in the files listed in `_defaults_`
            as they are loaded first. If you'd prefer a different order, you can add _self_ to the list of defaults. E.g.
            ```yaml
            _defaults_:
                - logging.yaml
                - _self_
                - database.yaml
            ```
            will result in database.yaml overriding fields in base.yaml resulting in the merged configuration:
            ```yaml
            logging:
                log_level: info
            database:
                host: localhost
                port: 5432
            ```

    Command line overrides:
        Command line arguments can be used to override configuration values. Overrides can
        be specified directly or by using the `_file_` keyword to load additional configuration files.

        Direct overrides:
        ```bash
        python script.py database.port=1234
        ```

        `_file_` override:
        The `_file_` override allows you to specify an additional YAML file to be merged
        into the existing configuration:
        ```bash
        python script.py _file_=extra_config.yaml
        ```

        `_file_` with key specification:
        To load a file and merge it at a specific key:
        ```bash
        python script.py _file_=extra_config.yaml@some.key
        ```

    Comparing `parse_omegaconf` with the `hydra` library:
        Hydra is a popular configuration management library built on top of OmegaConf. Which provides a large
        collection of additional features for managing complex configurations. Hydra is designed to be a
        comprehensive solution for configuration management, including support for running and managing
        multiple job executions. `parse_omegaconf` is a much simpler function that focuses solely on configuration
        loading and merging, without the additional features provided by Hydra. We wrote this function as
        a lightweight alternative which is less opinionated about how configurations should be structured, loaded,
        and used.

    Args:
        base_config_path (Optional[str]): The path to the base configuration file.
            If None, an empty configuration will be created.
        args (Optional[Sequence[str]]): A list of command line arguments to override
            configuration values. If None, `sys.argv[1:]` will be used.

    Returns:
        omegaconf.DictConfig: The final merged configuration.

    Raises:
        FileNotFoundError: If the specified base configuration file does not exist.
        ValueError: If a command line argument is not formatted as `key=value`.
        KeyError: If attempting to override a non-existent key without using the `+` prefix.
    """
    if base_config_path is not None:
        config = load_config_from_file(base_config_path)
    else:
        config = OmegaConf.create()

    if args is None:
        args = sys.argv[1:]

    while len(args) > 0:
        arg = args.pop(0)

        try:
            key, value = arg.split("=", maxsplit=1)
        except ValueError:
            raise ValueError(f"Invalid argument: {arg}. Must be formatted as key=value")

        if arg.startswith("_file_="):
            sub_key = None
            if "@" in value:
                value, sub_key = value.split("@")
            config = recursive_merge(
                config, load_config_from_file(value, at_key=sub_key)
            )
        else:
            try:
                value = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass

            if key.startswith("+"):
                key = key[1:]
            else:
                key_parts = key.split(".")
                sub_config = config
                err_msg = (
                    f"Cannot override value for key {key} in config to {value} using a command line argument"
                    f" as the key {key} does not already exist in the config. If you'd like to to add a"
                    f" key that does not already exist, please use the format +key=value rather than just key=value."
                )
                for key_part in key_parts[:-1]:
                    if key_part not in sub_config:
                        raise KeyError(err_msg)
                    sub_config = sub_config[key_part]

                if sub_config._get_node(key_parts[-1]) is None:
                    raise KeyError(err_msg)

            OmegaConf.update(config, key, value)

    return config
