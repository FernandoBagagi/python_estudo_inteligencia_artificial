import pytest

from mini_curso.exercicio_01 import Number
from mini_curso.exercicio_02_2 import _validate_not_empty, mean, median, mode


def numbers(*values: int | float) -> list[Number]:
    """Create Number instances for test data."""
    return [Number(value) for value in values]


def test_verify_empty_list_accepts_non_empty_lists() -> None:
    """Non-empty lists pass the validation helper."""
    _validate_not_empty(numbers(1))


def test_verify_empty_list_rejects_empty_lists() -> None:
    """The validation helper rejects empty lists."""
    with pytest.raises(ValueError, match='must not be empty'):
        _validate_not_empty([])


@pytest.mark.parametrize(
    ('function', 'values', 'expected'),
    [
        (mean, numbers(1, 2, 3, 4, 5), Number(3)),
        (mean, numbers(1, 2), Number(1.5)),
        (median, numbers(1, 2, 3, 4, 5), Number(3)),
        (median, numbers(1, 2, 3, 4), Number(2.5)),
        (mode, numbers(1, 2, 2, 3, 3), numbers(2, 3)),
        (mode, numbers(1, 1, 2, 3), numbers(1)),
    ],
)
def test_statistics_functions(function, values, expected) -> None:
    """Statistics functions return the expected result."""
    result = function(values)

    assert isinstance(result, Number) if function is not mode else all(
        isinstance(number, Number) for number in result
    )
    assert result == expected


@pytest.mark.parametrize('function', [mean, median, mode])
def test_statistics_functions_reject_empty_lists(function) -> None:
    """Statistics functions reject empty lists."""
    with pytest.raises(ValueError, match='must not be empty'):
        function([])


def test_exercise_example() -> None:
    """The exercise example returns a mean of 3.0."""
    sample_numbers = numbers(1, 2, 3, 4, 5)

    assert mean(sample_numbers) == 3
