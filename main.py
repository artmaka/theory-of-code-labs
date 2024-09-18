from itertools import combinations
import numpy as np


def REF_def(matrix: np.ndarray) -> np.ndarray:
    """
    Приводит матрицу к ступенчатому виду (Row Echelon Form).

    Args:
        matrix (np.ndarray): Входная бинарная матрица.

    Returns:
        np.ndarray: Матрица в ступенчатом виде.
    """
    matrix = matrix.copy()
    num_rows, num_cols = matrix.shape
    current_row = 0

    for col in range(num_cols):
        pivot_row = current_row
        while pivot_row < num_rows and matrix[pivot_row, col] == 0:
            pivot_row += 1

        if pivot_row == num_rows:
            continue

        if pivot_row != current_row:
            matrix[[pivot_row, current_row]] = matrix[[current_row, pivot_row]]

        for lower_row in range(current_row + 1, num_rows):
            if matrix[lower_row, col] == 1:
                matrix[lower_row] ^= matrix[current_row]

        current_row += 1
        if current_row == num_rows:
            break

    valid_rows = np.any(matrix, axis=1)
    return matrix[valid_rows]


def RREF_def(matrix: np.ndarray) -> np.ndarray:
    """
    Приводит матрицу к приведённому ступенчатому виду (Reduced Row Echelon Form).

    Args:
        matrix (np.ndarray): Входная бинарная матрица.

    Returns:
        np.ndarray: Приведённая ступенчатая матрица.
    """
    matrix = matrix.copy()
    num_rows, num_cols = matrix.shape

    for current_row in range(num_rows - 1, -1, -1):
        pivot_col = np.argmax(matrix[current_row] != 0)
        if matrix[current_row, pivot_col] == 0:
            continue

        for row_above in range(current_row):
            if matrix[row_above, pivot_col] == 1:
                matrix[row_above] ^= matrix[current_row]

    return matrix


def get_leading_columns(matrix: np.ndarray) -> np.ndarray:
    """
    Определяет индексы ведущих столбцов в матрице.

    Args:
        matrix (np.ndarray): Входная матрица.

    Returns:
        np.ndarray: Массив индексов ведущих столбцов.
    """
    pivot_col = []
    col = 0
    num_rows, num_cols = matrix.shape

    for i in range(num_rows):
        while col < num_cols and matrix[i, col] == 0:
            col += 1
        if col == num_cols:
            return np.array([])
        pivot_col.append(col)

    return np.array(pivot_col)


def remove_columns_from_matrix(matrix: np.ndarray, cols_to_delete: np.ndarray) -> np.ndarray:
    """
    Возвращает матрицу без ведущих столбцов.

    Args:
        matrix (np.ndarray): Входная матрица.
        cols_to_delete (np.ndarray): Массив индексов ведущих столбцов.

    Returns:
        np.ndarray: Матрица без ведущих столбцов.
    """
    columns_to_remove = set(cols_to_delete)
    transposed_matrix = matrix.T
    transposed_shortened_matrix = [row for idx, row in enumerate(transposed_matrix) if idx not in columns_to_remove]

    return np.array(transposed_shortened_matrix).T


def get_H_matrix(matrix: np.ndarray, leading: np.ndarray, num_rows: int) -> np.ndarray:
    """
    Формирует проверочную матрицу H.

    Args:
        matrix (np.ndarray): Сокращённая матрица X.
        leading (np.ndarray): Индексы ведущих столбцов.
        num_rows (int): Количество столбцов исходной матрицы.

    Returns:
        np.ndarray: Проверочная матрица H.
    """
    num_cols = matrix.shape[1]
    result = np.zeros((num_rows, num_cols), dtype=int)
    identity_matrix = np.eye(num_cols, dtype=int)
    short_index, unit_index, leading_index = 0, 0, 0
    num_leading_cols = len(leading)

    for i in range(num_rows):
        if leading_index < num_leading_cols and i == leading[leading_index]:
            result[i] = matrix[short_index]
            short_index += 1
            leading_index += 1
        else:
            result[i] = identity_matrix[unit_index]
            unit_index += 1

    return result


def compute_linear_combinations(matrix: np.ndarray) -> np.ndarray:
    """
    Генерирует все линейные комбинации строк матрицы.

    Args:
        matrix (np.ndarray): Входная матрица.

    Returns:
        np.ndarray: Массив всех линейных комбинаций.
    """
    num_rows = matrix.shape[0]
    all_combinations = set()

    for row in range(2, num_rows + 1):
        for comb in combinations(range(num_rows), row):
            combination = np.sum(matrix[list(comb)], axis=0) % 2
            all_combinations.add(tuple(combination))

    result_array = np.array(list(all_combinations), dtype=int)
    return result_array


def generate_all_codewords(num: int) -> np.ndarray:
    """
    Генерирует все двоичные слова заданной длины.

    Args:
        num (int): Длина двоичного слова.

    Returns:
        np.ndarray: Массив всех двоичных слов.
    """
    get_combinations = []
    for i in range(2 ** num):
        combination = [(i >> j) & 1 for j in range(num)]
        get_combinations.append(combination)
    return np.array(get_combinations)


def encode_binary_words(G: np.ndarray, all_words: np.ndarray) -> np.ndarray:
    """
    Умножает все двоичные слова на порождающую матрицу G.

    Args:
        G (np.ndarray): Порождающая матрица G.
        all_words (np.ndarray): Все двоичные слова длины k.

    Returns:
        np.ndarray: Массив всех кодовых слов.
    """
    return np.dot(all_words, G) % 2


def Hamming_weight(word: np.ndarray) -> int:
    """
    Вычисляет вес Хэмминга для двоичного слова.

    Args:
        word (np.ndarray): Входное двоичное слово.

    Returns:
        int: Вес Хэмминга.
    """
    return sum(word)


def Hamming_distance(words: np.ndarray) -> int:
    """
    Вычисляет минимальное кодовое расстояние (расстояние Хэмминга) среди кодовых слов.

    Args:
        words (np.ndarray): Массив кодовых слов.

    Returns:
        int: Минимальное кодовое расстояние.
    """
    result = float('inf')
    for word in words:
        result = min(result, Hamming_weight(word))
    return result


if __name__ == "__main__":
    default_matrix = np.array([
        [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
        [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    ])

    step_matrix = REF_def(default_matrix)
    print("\nСтупенчатая матрица G", step_matrix, sep='\n')

    reduced_step_matrix = RREF_def(step_matrix)
    k, n = len(reduced_step_matrix), len(reduced_step_matrix[0])
    print("\nПриведенная ступенчатая матрица G*", reduced_step_matrix, sep='\n')

    leading_cols = get_leading_columns(reduced_step_matrix)
    print("\nИндексы ведущих столбцов", leading_cols, sep='\n')

    shortened_matrix = remove_columns_from_matrix(reduced_step_matrix, leading_cols)
    print("\nМатрица X после удаления ведущих столбцов", shortened_matrix, sep='\n')

    h_matrix = get_H_matrix(shortened_matrix, leading_cols, n)
    print("\nПроверочная матрица H", h_matrix, sep='\n')
