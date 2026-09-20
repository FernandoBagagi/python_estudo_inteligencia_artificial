import pytest

from mini_curso.exercicio_01 import Number
from mini_curso.exercicio_02_1 import (
    comprehension_even_numbers,
    filter_even_numbers,
    is_even,
    to_number,
)


def test_to_number_converts_numeric_values() -> None:
    """The conversion helper always returns a Number instance."""
    result = to_number(5)

    assert isinstance(result, Number)
    assert result == 5


@pytest.mark.parametrize(
    ('value', 'expected'),
    [(Number(2), True), (Number(3), False), (Number(4.0), True)],
)
def test_is_even(value: Number, expected: bool) -> None:
    """The predicate identifies even and odd Number values."""
    assert is_even(value) is expected


@pytest.mark.parametrize('function', [filter_even_numbers, comprehension_even_numbers])
def test_even_number_solutions(function) -> None:
    """Both didactic implementations return the same even values."""
    result = function([1, 2, 3, 4, 5])

    assert result == [Number(2), Number(4)]
    assert all(isinstance(number, Number) for number in result)


@pytest.mark.parametrize('function', [filter_even_numbers, comprehension_even_numbers])
def test_even_number_solutions_handle_empty_input(function) -> None:
    """Both implementations return an empty list for empty input."""
    assert function([]) == []


def test_main_example() -> None:
    """Both approaches produce the expected result for the exercise example."""
    sample_numbers = [1, 2, 3, 4, 5]
    expected = [Number(2), Number(4)]

    assert filter_even_numbers(sample_numbers) == expected
    assert comprehension_even_numbers(sample_numbers) == expected
