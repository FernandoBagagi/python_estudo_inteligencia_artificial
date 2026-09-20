import pytest

from mini_curso.exercicio_01 import Number
from mini_curso.exercicio_02_3 import (
    difference,
    difference_builtin,
    intersection,
    intersection_builtin,
    symmetric_difference,
    symmetric_difference_builtin,
    union,
    union_builtin,
)


def number_set(*values: int) -> set[Number]:
    """Create a set of Number instances for test data."""
    return {Number(value) for value in values}


@pytest.mark.parametrize(
    ('manual', 'builtin', 'expected'),
    [
        (intersection, intersection_builtin, number_set(2, 3)),
        (union, union_builtin, number_set(1, 2, 3, 4, 5, 8, 14)),
        (difference, difference_builtin, number_set(1, 4, 5)),
        (
            symmetric_difference,
            symmetric_difference_builtin,
            number_set(1, 4, 5, 8, 14),
        ),
    ],
)
def test_manual_and_builtin_set_operations(manual, builtin, expected) -> None:
    """Manual and built-in implementations return the same result."""
    first = number_set(1, 2, 3, 4, 5)
    second = number_set(2, 3, 8, 14)

    assert manual(first, second) == expected
    assert builtin(first, second) == expected


@pytest.mark.parametrize(
    'operation',
    [
        (intersection, intersection_builtin),
        (union, union_builtin),
        (difference, difference_builtin),
        (symmetric_difference, symmetric_difference_builtin),
    ],
)
def test_set_operations_with_empty_sets(operation) -> None:
    """Set implementations handle empty sets consistently."""
    manual, builtin = operation

    assert manual(set(), number_set(1, 2)) == builtin(set(), number_set(1, 2))
