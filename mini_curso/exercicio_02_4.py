"""Sort a list of numbers using four classic algorithms."""

from mini_curso.exercicio_01 import Number


def bubble_sort(numbers: list[Number]) -> list[Number]:
    """Return a sorted copy of ``numbers`` using bubble sort."""
    if not numbers:
        return []

    sorted_numbers = list(numbers)
    size = len(sorted_numbers)

    for pass_index in range(size - 1):
        swapped = False
        for index in range(size - 1 - pass_index):
            if sorted_numbers[index] > sorted_numbers[index + 1]:
                sorted_numbers[index], sorted_numbers[index + 1] = (
                    sorted_numbers[index + 1],
                    sorted_numbers[index],
                )
                swapped = True

        if not swapped:
            break

    return sorted_numbers


def selection_sort(numbers: list[Number]) -> list[Number]:
    """Return a sorted copy of ``numbers`` using selection sort."""
    sorted_numbers = list(numbers)
    size = len(sorted_numbers)

    for index in range(size):
        smallest_index = index
        for candidate_index in range(index + 1, size):
            if sorted_numbers[candidate_index] < sorted_numbers[smallest_index]:
                smallest_index = candidate_index

        sorted_numbers[index], sorted_numbers[smallest_index] = (
            sorted_numbers[smallest_index],
            sorted_numbers[index],
        )

    return sorted_numbers


def insertion_sort(numbers: list[Number]) -> list[Number]:
    """Return a sorted copy of ``numbers`` using insertion sort."""
    sorted_numbers = list(numbers)

    for index in range(1, len(sorted_numbers)):
        value = sorted_numbers[index]
        previous_index = index - 1

        # Move larger values one position to the right.
        while previous_index >= 0 and sorted_numbers[previous_index] > value:
            sorted_numbers[previous_index + 1] = sorted_numbers[previous_index]
            previous_index -= 1

        sorted_numbers[previous_index + 1] = value

    return sorted_numbers


def quick_sort(numbers: list[Number]) -> list[Number]:
    """Return a sorted copy of ``numbers`` using recursive quick sort."""
    # Lists with zero or one element are already sorted.
    if len(numbers) <= 1:
        return list(numbers)

    # Choose the middle value as the pivot.
    pivot = numbers[len(numbers) // 2]
    left = [number for number in numbers if number < pivot]
    equal = [number for number in numbers if number == pivot]
    right = [number for number in numbers if number > pivot]

    # Recursively sort both partitions and combine the results.
    return quick_sort(left) + equal + quick_sort(right)
