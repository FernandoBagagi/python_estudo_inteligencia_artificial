import pytest

from mini_curso.exercicio_01 import Number


@pytest.mark.parametrize(
    ('method_name', 'number', 'other', 'expected'),
    [
        ('plus', 2, 3, 5),
        ('subtract', 5, 3, 2),
        ('multiply', 3, 4, 12),
        ('divide', 15, 3, 5),
        ('floor_divide', 15, 4, 3),
        ('modulo', 15, 4, 3),
        ('power', 2, 3, 8),
    ],
)
def test_arithmetic_methods_return_a_number(
    method_name: str, number: float, other: float, expected: float
) -> None:
    """Arithmetic methods return the expected value as a Number."""
    result = getattr(Number(number), method_name)(other)

    assert isinstance(result, Number)
    assert result == expected


@pytest.mark.parametrize(
    ('operator', 'number', 'other', 'expected'),
    [
        (lambda number, other: number + other, 2, 3, 5),
        (lambda number, other: number - other, 5, 3, 2),
        (lambda number, other: number * other, 3, 4, 12),
        (lambda number, other: number / other, 15, 3, 5),
        (lambda number, other: number // other, 15, 4, 3),
        (lambda number, other: number % other, 15, 4, 3),
        (lambda number, other: number**other, 2, 3, 8),
    ],
)
def test_python_operators_return_a_number(operator, number, other, expected) -> None:
    """Overloaded Python operators return the expected Number value."""
    result = operator(Number(number), other)

    assert isinstance(result, Number)
    assert result == expected


def test_operations_can_be_chained_without_mutating_the_original() -> None:
    """Chained operations create a new value and preserve the original."""
    number = Number(5.1)

    result = number.plus(1).multiply(2)

    assert result == 12.2
    assert number == 5.1


def test_division_by_zero_raises_error() -> None:
    """Division by zero raises Python's standard exception."""
    number = Number(10)
    with pytest.raises(ZeroDivisionError):
        number.divide(0)
