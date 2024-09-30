from itertools import combinations
import numpy as np
import random


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
        all_words (np.ndarray): Массив всех двоичных слов.

    Returns:
        np.ndarray: Массив закодированных слов.
    """
    return (all_words @ G) % 2


def generate_syndrome_table(H: np.ndarray, error_weight: int) -> dict:
    """
    Генерирует таблицу синдромов для кодов с фиксированным весом ошибки.

    Args:
        H (np.ndarray): Проверочная матрица.
        error_weight (int): Максимальный вес ошибки.

    Returns:
        dict: Таблица синдромов с индексами ошибок.
    """
    n = H.shape[1]
    syndrome_table = {}
    for error in range(1, error_weight + 1):
        for error_indices in combinations(range(n), error):
            error_vector = np.zeros(n, dtype=int)
            for index in error_indices:
                error_vector[index] = 1
            syndrome = error_vector @ H % 2
            syndrome_table[tuple(map(int, syndrome))] = tuple(error_indices)

    return syndrome_table


if __name__ == "__main__":
    # Пример работы с первой матрицей
    s_matrix = np.array([[1, 0, 0, 1, 0, 1, 1],
                         [1, 1, 0, 0, 0, 0, 1],
                         [0, 0, 1, 1, 0, 0, 1],
                         [1, 0, 1, 0, 1, 0, 1],
                         [0, 0, 1, 1, 1, 1, 0]])
    G = RREF_def(REF_def(s_matrix))
    print("Порождающая матрица G:\n", G)

    G_standard = REF_def(G)
    print("Порождающая матрица G в стандартном виде:\n", G_standard)

    leading_cols = get_leading_columns(G_standard)
    H_matrix = get_H_matrix(G_standard, leading_cols, G_standard.shape[1])
    print("Проверочная матрица H:\n", H_matrix)

    # Генерация таблицы синдромов
    syndrome_table = generate_syndrome_table(H_matrix, 1)
    print("Таблица синдромов:\n", syndrome_table)

    u = np.array([1, 0, 0, 1])
    v = u @ G_standard % 2
    print("Кодовое слово длины k = 4:\n", v)

    # Добавляем ошибку в случайной позиции
    error = np.zeros(7, dtype=int)
    error[random.randint(0, 6)] = 1
    v_with_error = (v + error) % 2
    print("Принятое с ошибкой слово:\n", v_with_error)

    # Вычисление синдрома
    syndrome = v_with_error @ H_matrix % 2
    print("Синдром принятого сообщения:\n", syndrome)

    # Исправление ошибки
    if tuple(syndrome) in syndrome_table:
        error_position = syndrome_table[tuple(syndrome)][0]
        error[error_position] = 1
        corrected_message = (v_with_error + error) % 2
        print("Исправленное сообщение:\n", corrected_message)
    else:
        print("Ошибка не найдена в таблице синдромов.")
