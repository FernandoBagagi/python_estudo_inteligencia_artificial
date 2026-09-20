"""Filter even numbers using two different Python approaches.

This exercise intentionally keeps both implementations so they can be
compared: one uses ``map`` and ``filter`` and the other uses a list
comprehension.
"""

from mini_curso.exercicio_01 import Number


def to_number(number: int | float | Number) -> Number:
    """Convert ``number`` to a :class:`~mini_curso.exercicio_01.Number`."""
    return Number(number)


def is_even(number: Number) -> bool:
    """Return ``True`` when ``number`` is even."""
    return number.modulo(2) == 0


def filter_even_numbers(
    numbers: list[int] | list[float] | list[Number],
) -> list[Number]:
    """Return even numbers using ``map`` followed by ``filter``."""
    converted_numbers = map(to_number, numbers)
    return list(filter(is_even, converted_numbers))


def comprehension_even_numbers(
    numbers: list[int] | list[float] | list[Number],
) -> list[Number]:
    """Return even numbers using a list comprehension."""
    return [number for number in (Number(n) for n in numbers) if number % 2 == 0]
