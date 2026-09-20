from itertools import permutations
from time import perf_counter

import pytest

from mini_curso.exercicio_01 import Number
from mini_curso.exercicio_02_4 import (
    bubble_sort,
    insertion_sort,
    quick_sort,
    selection_sort,
)


def numbers(*values: int | float) -> list[Number]:
    """Create Number instances for test data."""
    return [Number(value) for value in values]


@pytest.mark.parametrize(
    'sort_function', [bubble_sort, selection_sort, insertion_sort, quick_sort]
)
def test_sorting_functions_sort_the_exercise_example(sort_function) -> None:
    """Every sorting algorithm orders the exercise example correctly."""
    assert sort_function(numbers(3, 2, 5, 1, 4)) == numbers(1, 2, 3, 4, 5)


@pytest.mark.parametrize(
    'sort_function', [bubble_sort, selection_sort, insertion_sort, quick_sort]
)
@pytest.mark.parametrize(
    'values', [[], [1], [2, 1, 2, 3, 1], [-2, 0, 1, -1]]
)
def test_sorting_functions_handle_varied_input(sort_function, values) -> None:
    """Every sorting algorithm handles empty, duplicate, and negative values."""
    result = sort_function(numbers(*values))

    assert result == numbers(*sorted(values))


@pytest.mark.parametrize(
    'sort_function', [bubble_sort, selection_sort, insertion_sort, quick_sort]
)
def test_sorting_functions_do_not_mutate_input(sort_function) -> None:
    """Every sorting algorithm returns a new list."""
    original = numbers(3, 1, 2)

    result = sort_function(original)

    assert result == numbers(1, 2, 3)
    assert original == numbers(3, 1, 2)
    assert result is not original


@pytest.mark.parametrize('values', list(permutations([0, 1, 2, 3])))
@pytest.mark.parametrize(
    'sort_function', [bubble_sort, selection_sort, insertion_sort, quick_sort]
)
def test_sorting_functions_exhaustive_for_small_vectors(sort_function, values) -> None:
    """Every algorithm sorts every permutation of a small vector."""
    result = sort_function(numbers(*values))

    assert result == numbers(0, 1, 2, 3)


@pytest.mark.parametrize(
    'sort_function', [bubble_sort, selection_sort, insertion_sort, quick_sort]
)
def test_sorting_performance(sort_function) -> None:
    """Measure each algorithm without making timing a pass/fail condition."""
    values = numbers(*range(1000, 0, -1))

    start = perf_counter()
    result = sort_function(values)
    elapsed = perf_counter() - start

    assert result == numbers(*range(1, 1001))
    print(f'{sort_function.__name__}: {elapsed:.6f}s')
