"""Perform set operations with manual and built-in implementations."""

from mini_curso.exercicio_01 import Number


def intersection(first: set[Number], second: set[Number]) -> set[Number]:
    """Return the values present in both sets using iteration."""
    if not first or not second:
        return set()
    return {number for number in first if number in second}


def intersection_builtin(first: set[Number], second: set[Number]) -> set[Number]:
    """Return the intersection
    Python can perform set intersection natively with the ``&`` operator
    on classes that implement ``__hash__`` and ``__eq__``."""

    return first & second


def union(first: set[Number], second: set[Number]) -> set[Number]:
    """Return the union using a copied set and ``update``."""
    result = set(first)
    result.update(second)
    return result


def union_builtin(first: set[Number], second: set[Number]) -> set[Number]:
    """Return the union
    Python can perform set union natively with the ``|`` operator
    on classes that implement ``__hash__`` and ``__eq__``."""

    return first | second


def difference(first: set[Number], second: set[Number]) -> set[Number]:
    """Return values in ``first`` that are not in ``second``."""
    return {number for number in first if number not in second}


def difference_builtin(first: set[Number], second: set[Number]) -> set[Number]:
    """Return the difference
    Python can perform set difference natively with the ``-`` operator
    on classes that implement ``__hash__`` and ``__eq__``."""

    return first - second


def symmetric_difference(first: set[Number], second: set[Number]) -> set[Number]:
    """Return values present in exactly one of the two sets."""
    result = {number for number in first if number not in second}
    result.update(number for number in second if number not in first)
    return result


def symmetric_difference_builtin(
    first: set[Number], second: set[Number]
) -> set[Number]:
    """Return the symmetric difference
    Python can perform symmetric difference natively with the ``^`` operator
    on classes that implement ``__hash__`` and ``__eq__``."""

    return first ^ second
