"""Define a small numeric value object for arithmetic exercises."""

from __future__ import annotations


class Number:
    """Represent a numeric value with chainable arithmetic operations.

    Arithmetic methods return new ``Number`` instances, so operations can be
    chained without changing the original value.

    :param value: The numeric value to wrap.
    :ivar value: The wrapped value, represented as a ``float``.
    """

    def __init__(self, value: int | float | Number):
        self.value = value.value if isinstance(value, Number) else float(value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (int, float)):
            return self.value == Number(other).value
        if isinstance(other, Number):
            return self.value == other.value
        return False

    def plus(self, other: int | float | Number) -> Number:
        return Number(self.value + Number(other).value)

    def subtract(self, other: int | float | Number) -> Number:
        return Number(self.value - Number(other).value)

    def multiply(self, other: int | float | Number) -> Number:
        return Number(self.value * Number(other).value)

    def divide(self, other: int | float | Number) -> Number:
        return Number(self.value / Number(other).value)

    def floor_divide(self, other: int | float | Number) -> Number:
        return Number(self.value // Number(other).value)

    def modulo(self, other: int | float | Number) -> Number:
        return Number(self.value % Number(other).value)

    def power(self, other: int | float | Number) -> Number:
        return Number(self.value ** Number(other).value)

    __add__ = plus
    __sub__ = subtract
    __mul__ = multiply
    __truediv__ = divide
    __floordiv__ = floor_divide
    __mod__ = modulo
    __pow__ = power


if __name__ == '__main__':
    number = Number(5.2) + 1 - Number(2.2)
    print(number.value)
