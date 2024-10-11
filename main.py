import numpy as np
import itertools
import random


def generate_hamming_h_matrix(r: int) -> np.ndarray:
    """
    Генерирует проверочную матрицу кода Хэмминга.

    Args:
        r (int): Параметр кода.

    Returns:
        np.ndarray: Проверочная матрица H.
    """
    n = 2 ** r - 1  # Число строк и столбцов
    res = []
    cur_r = r - 1
    for i in range(n, 0, -1):
        if i != 2 ** cur_r:
            res.append(list(map(int, f"{i:0{r}b}")))
        else:
            cur_r -= 1

    identity_matrix = np.eye(r, dtype=int)  # Единичная матрица
    H = np.vstack((res, identity_matrix))  # Объединяем верхнюю часть и единичную матрицу

    return H


def H_to_G(H: np.ndarray, r: int) -> np.ndarray:
    """
    Строит порождающую матрицу на основе проверочной матрицы.

    Args:
        H (np.ndarray): Проверочная матрица.
        r (int): Параметр кода.

    Returns:
        np.ndarray: Порождающая матрица G.
    """
    k = 2 ** r - r - 1
    G = np.hstack((np.eye(k, dtype=int), H[:k]))  # Объединяем единичную матрицу с H
    return G


def generate_syndrome_table(H: np.ndarray, error_weight: int) -> dict:
    """
    Генерирует таблицу синдромов.

    Args:
        H (np.ndarray): Проверочная матрица.
        error_weight (int): Максимальный вес ошибок.

    Returns:
        dict: Таблица синдромов.
    """
    n = H.shape[0]
    syndrome_table = {}
    for error in range(1, error_weight + 1):
        for error_indices in itertools.combinations(range(n), error):
            error_vector = np.zeros(n, dtype=int)
            for index in error_indices:
                error_vector[index] = 1
            syndrome = error_vector @ H % 2
            syndrome_table[tuple(map(int, syndrome))] = tuple(error_indices)

    return syndrome_table


def hamming_correction_test(G: np.ndarray, H: np.ndarray, syndrome_table: dict, error_degree: int, u: np.ndarray):
    """
    Тестирование исправления ошибок кода Хэмминга.

    Args:
        G (np.ndarray): Порождающая матрица.
        H (np.ndarray): Проверочная матрица.
        syndrome_table (dict): Таблица синдромов.
        error_degree (int): Число допустимых ошибок.
        u (np.ndarray): Исходное сообщение.
    """
    print("Кодовое слово (u):", u)

    v = u @ G % 2  # Генерация кодового слова
    print("Отправленное кодовое слово (v):", v)

    # Допуск ошибки
    error_indices = random.sample(range(v.shape[0]), error_degree)
    error = np.zeros(v.shape[0], dtype=int)
    for index in error_indices:
        error[index] = 1
    print("Допущенная ошибка:", error)

    received_v = (v + error) % 2  # Принятое слово с ошибкой
    print("Принятое с ошибкой слово:", received_v)

    syndrome = received_v @ H % 2  # Вычисляем синдром
    print("Синдром принятого сообщения:", syndrome)
    if np.any(syndrome):
        print("Обнаружена ошибка!")

    if tuple(syndrome) in syndrome_table:
        correction_indices = syndrome_table[tuple(syndrome)]
        for index in correction_indices:
            received_v[index] = (received_v[index] + 1) % 2  # Корректируем ошибку
        print("Исправленное сообщение:", received_v)

        if np.array_equal(v, received_v):
            print("Ошибка была исправлена успешно!")
        else:
            print("Ошибка не была исправлена корректно.")
    else:
        print("Синдрома нет в таблице, ошибка не исправлена.")


def expand_G_matrix(G: np.ndarray) -> np.ndarray:
    """
    Расширяет порождающую матрицу.

    Args:
        G (np.ndarray): Порождающая матрица.

    Returns:
        np.ndarray: Расширенная порождающая матрица.
    """
    col = np.zeros((G.shape[0], 1), dtype=int)
    for i in range(G.shape[0]):
        if sum(G[i]) % 2 == 1:
            col[i] = 1
    return np.hstack((G, col))


if __name__ == '__main__':

    r = 2
    H = generate_hamming_h_matrix(r)
    print("Возьмем r = 2, получим проверочную матрицу следующего вида:")
    print(H)

    G = H_to_G(H, r)
    print("\nПроверим функцию и построим порождающую матрицу:")
    print(G)

    print("\n3.2\n")
    syndrome_table = generate_syndrome_table(H, 1)
    print("Сформируем таблицу синдромов для всех единичных ошибок:")
    print(syndrome_table)

    print(
        "\nПроведем исследование кода Хэмминга для ошибки кратности 1. Передаем сообщение длины 2^r - r - 1 = 2^2 - 2 - 1 = 1: 1 0 0 1")
    hamming_correction_test(G, H, syndrome_table, 1, np.array([1]))

    print("\nТеперь попробуем допустить двухкратную ошибку:")
    hamming_correction_test(G, H, syndrome_table, 2, np.array([1]))

    print("\nПопробуем трехкратную ошибку:")
    hamming_correction_test(G, H, syndrome_table, 3, np.array([1]))

    print("\nАналогично, проведем исследование для r = 3:")
    r = 3
    H = generate_hamming_h_matrix(r)
    print(H)

    G = H_to_G(H, r)
    print("\nПересчитаем G:")
    print(G)

    syndrome_table = generate_syndrome_table(H, 1)
    print("\nСоставим заново таблицу синдромов:")
    print(syndrome_table)

    print("\nПередаем слово длины 4: 1 0 0 1. Сначала допустим однократную ошибку:")
    hamming_correction_test(G, H, syndrome_table, 1, np.array([1, 0, 0, 1]))

    print("\nТеперь для двухкратной ошибки:")
    hamming_correction_test(G, H, syndrome_table, 2, np.array([1, 0, 0, 1]))

    print("\nИ для трехкратной ошибки:")
    hamming_correction_test(G, H, syndrome_table, 3, np.array([1, 0, 0, 1]))

    print("\nРезультаты аналогичны r = 2. Теперь проведем исследование для r = 4:")
    r = 4
    H = generate_hamming_h_matrix(r)
    print(H)

    G = H_to_G(H, r)
    print("\nПересчитаем G:")
    print(G)

    syndrome_table = generate_syndrome_table(H, 1)
    print("\nЗаново составим таблицу синдромов:")
    print(syndrome_table)

    print("\nПередаем сообщение длины 11: 0 0 1 0 1 1 0 0 1 1 1. Допустим однократную ошибку:")
    hamming_correction_test(G, H, syndrome_table, 1, np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1]))

    print("\nТеперь допустим двухкратную ошибку:")
    hamming_correction_test(G, H, syndrome_table, 2, np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1]))

    print("\nИ, наконец, трехкратную ошибку:")
    hamming_correction_test(G, H, syndrome_table, 3, np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1]))

    print(
        "\nПолучили результаты аналогичные r = 2 и r = 3. Код Хэмминга позволяет либо исправлять однократные ошибки, либо обнаруживать двухкратные.")

