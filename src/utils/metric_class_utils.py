"""Utility functions to import metric classes."""

from typing import Any, List
import importlib


def import_class(name: str) -> Any:
    """
    Function to import the class given its name.
    Args:
        name (string): name of the class to import.
    Returns:
        class object.
    """
    components = name.split(".")
    cls_name = components[-1]
    module_name = ".".join(components[:-1])
    module = importlib.import_module(module_name)
    cls_ = getattr(module, cls_name)
    return cls_


def get_metric_class_list(metric_str_list: List[str]) -> List[Any]:
    """
    Function to import the classes given their names in a list.
    Args:
        metric_str_list (list): list of strings containing the names of the classes to import.
    Returns:
        list of class objects.
    """
    metric_cls_list = []
    for metric_str in metric_str_list:
        metric_cls_list.append(import_class(metric_str))
    return metric_cls_list
