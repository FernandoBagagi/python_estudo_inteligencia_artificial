"""Calculate the mean, median, and mode of a list of numbers."""

from mini_curso.exercicio_01 import Number


def _validate_not_empty(numbers: list[Number]) -> None:
    """Raise ``ValueError`` when ``numbers`` is empty."""
    if not numbers:
        raise ValueError('numbers must not be empty')


def mean(numbers: list[Number]) -> Number:
    """Return the arithmetic mean of ``numbers``."""
    _validate_not_empty(numbers)

    numbers_sum = sum(numbers, start=Number(0))
    return numbers_sum / len(numbers)


def median(numbers: list[Number]) -> Number:
    """Return the middle value of the ordered ``numbers`` list."""
    _validate_not_empty(numbers)

    sorted_numbers = sorted(numbers)

    middle_index = len(sorted_numbers) // 2

    middle_value = sorted_numbers[middle_index]

    if len(sorted_numbers) % 2 == 0:
        # Uses the bitwise operator to access the symmetric index (index_middle - 1)
        middle_value += sorted_numbers[~middle_index]
        middle_value /= 2

    return middle_value


def mode(numbers: list[Number]) -> list[Number]:
    """Return all values with the highest frequency in ``numbers``."""
    _validate_not_empty(numbers)

    occurrences_by_number: dict[Number, int] = {}

    for number in numbers:
        occurrences_by_number[number] = occurrences_by_number.get(number, 0) + 1

    highest_occurrence_count = max(occurrences_by_number.values())

    return [
        number
        for number, occurrence_count in occurrences_by_number.items()
        if occurrence_count == highest_occurrence_count
    ]
