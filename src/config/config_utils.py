import yaml
import os
from functools import reduce


class Config:
    """
    Class to load the configuration file(s) and to get the value associated with the key using dot notation as the key.
    """

    def __init__(
        self,
        config_path: str,
        name_list: list[str] = None,
        not_to_read: list[str] = None,
    ):
        """
        Initialization of the Config class with the attribute of config's file path and the loading of it.
        Args:
            config_path (string): mandatory path to the configuration file or to the folder containing the configuration files.
            name_list (list[string]): optional list of configuration file names to read.
            not_to_read (list[string]): optional list of configuration file names to exclude.
        """
        self.config_path = config_path
        self.name_list = name_list
        self.not_to_read = not_to_read
        self._config = self._load_config()

    def _load_config(self) -> dict:
        """
        Method to load as dict the config file(s) from the config's file path or from the list of config file(s) name.
        Returns:
            a dictionary of the config file(s).
        """
        # if there is not the name_list but the config file name is in the config_path
        if (self.name_list is None) & (".yml" in self.config_path):
            self.name_list = [self.config_path]
        # if there is not the name_list and the config file name is not in the config_path
        elif (self.name_list is None) & (".yml" not in self.config_path):
            self.name_list = [
                self.config_path + c
                for c in os.listdir(self.config_path)
                if ".yml" in c
            ]
        # if there are name_list values
        else:
            self.name_list = [
                self.config_path + c for c in self.name_list if ".yml" in c
            ]

        # if there is a not_to_read list, it excludes some files
        if self.not_to_read is not None:
            self.name_list = [
                c for c in self.name_list if c.split(r"/")[-1] not in self.not_to_read
            ]

        # Reading all the config files
        config_union_dict = {}
        all_keys = []
        for c in self.name_list:
            with open(c, "r") as f:
                config_dict = yaml.safe_load(f)
                some_keys = list(config_dict.keys())
                duplicated_keys = [k for k in some_keys if k in all_keys]
                if duplicated_keys == []:
                    config_union_dict.update(config_dict)
                    all_keys = list(config_union_dict.keys())
                else:
                    raise ValueError(
                        f"Duplicate keys found in config files: {duplicated_keys}"
                    )
        return config_union_dict

    def print_config(self):
        """
        Method to print the read config.
        """
        print(yaml.dump(self._config))

    def check_config_value(self, keys: str) -> bool:
        """
        Method to check if a combination of keys is present in the config.
        Args:
            keys (string): string of keys separated by ".
        Returns:
            boolean: True if the combination of keys is present in the config, False otherwise.
        """
        config_dict = self._config
        keys = keys.split(".")
        for key in keys:
            if key not in config_dict.keys():
                return False
            config_dict = config_dict[key]
        return True

    def get_config_value(self, keys: str):
        """
        Method that returns the value associated with the key using dot notation as the key.
        Args:
            keys (string): string of keys separated by ".".
        Returns:
            a value associated with the key.
        Raises:
            KeyError: if the key is not present in the config.
        """
        try:
            return reduce(lambda c, k: c[k], keys.split("."), self._config)
        except KeyError:
            raise KeyError("Key {} not present in config file".format(keys))
