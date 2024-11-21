import numpy as np
import random

# Расширенный код Голея
B = np.array([
    [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
    [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
    [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
    [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
    [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
    [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
    [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])

def G_H(B):
    """
    Генерация порождающей (G) и проверочной (H) матриц для расширенного кода Голея.

    Параметры:
        B (np.ndarray): Бинарная матрица, определяющая код Голея.

    Возвращает:
        tuple: Порождающая матрица G и проверочная матрица H.
    """
    G = np.hstack((np.eye(12, dtype=int), B))
    H = np.vstack((np.eye(12, dtype=int), B))
    return G, H

def create_errors(word, G, count):
    """
    Внесение ошибок в передаваемое кодовое слово.

    Параметры:
        word (np.ndarray): Исходное кодовое слово.
        G (np.ndarray): Порождающая матрица.
        count (int): Количество ошибок для внесения.

    Возвращает:
        np.ndarray: Кодовое слово с ошибками.
    """
    crypt_word = word @ G % 2
    err_positions = random.sample(range(crypt_word.shape[0]), count)
    error = np.zeros(crypt_word.shape[0], dtype=int)
    for index in err_positions:
        error[index] = 1
    word_with_mistake = (crypt_word + error) % 2
    return word_with_mistake

def find_mistake(word_with_mistake, H, B):
    """
    Обнаружение и локализация ошибок в принятом кодовом слове с использованием проверочной матрицы.

    Параметры:
        word_with_mistake (np.ndarray): Принятое кодовое слово с ошибками.
        H (np.ndarray): Проверочная матрица.
        B (np.ndarray): Матрица, определяющая код Голея.

    Возвращает:
        np.ndarray или None: Выявленный паттерн ошибок или None, если ошибки не поддаются исправлению.
    """
    s = word_with_mistake @ H % 2
    mistakes = None
    if sum(s) <= 3:
        mistakes = np.hstack((s, np.zeros(len(s), dtype=int)))
    else:
        for i in range(len(B)):
            temp = (s + B[i]) % 2
            if sum(temp) <= 2:
                mistake_index = np.zeros(len(s), dtype=int)
                mistake_index[i] = 1
                mistakes = np.hstack((temp, mistake_index))
    return mistakes

def correct_mistake(true_word, word_with_mistake, H, B, G):
    """
    Исправление ошибок в принятом кодовом слове.

    Параметры:
        true_word (np.ndarray): Исходное кодовое слово.
        word_with_mistake (np.ndarray): Принятое кодовое слово с ошибками.
        H (np.ndarray): Проверочная матрица.
        B (np.ndarray): Матрица, определяющая код Голея.
        G (np.ndarray): Порождающая матрица.
    """
    mistakes = find_mistake(word_with_mistake, H, B)
    if mistakes is None:
        print("Обнаружена ошибка, исправление невозможно.")
        return
    corrected_word = (word_with_mistake + mistakes) % 2
    word = true_word @ G % 2
    if not np.array_equal(word, corrected_word):
        print("Ошибка декодирования!")

def G_RM(r, m):
    """
    Генерация порождающей матрицы для кода Рида-Маллера RM(r, m).

    Параметры:
        r (int): Порядок кода.
        m (int): Длина кода.

    Возвращает:
        np.ndarray: Порождающая матрица для кода Рида-Маллера.
    """
    if 0 < r < m:
        leftup = G_RM(r, m - 1)
        rightlow = G_RM(r - 1, m - 1)
        return np.hstack([np.vstack([leftup, np.zeros((len(rightlow), len(leftup.T)), int)]), 
                          np.vstack([leftup, rightlow])])
    elif r == 0:
        return np.ones((1, 2 ** m), dtype=int)
    elif r == m:
        up = G_RM(m - 1, m)
        low = np.zeros((1, 2 ** m), dtype=int)
        low[0][len(low.T) - 1] = 1
        return np.vstack([up, low])

def H_RM(i, m):
    """
    Генерация проверочной матрицы для кода Рида-Маллера RM(r, m).

    Параметры:
        i (int): Текущий уровень построения по методу произведения Кронекера.
        m (int): Длина кода.

    Возвращает:
        np.ndarray: Проверочная матрица для RM(i, m).
    """
    H = np.array([[1, 1], [1, -1]])
    result = np.kron(np.eye(2 ** (m - i)), H)
    result = np.kron(result, np.eye(2 ** (i - 1)))
    return result

def research_with_RM(word, G, count, m):
    """
    Исследование кода Рида-Маллера RM(r, m) в условиях ошибок.

    Параметры:
        word (np.ndarray): Исходное кодовое слово.
        G (np.ndarray): Порождающая матрица для RM(r, m).
        count (int): Количество вносимых ошибок.
        m (int): Длина кода.
    """
    word_with_mistake = create_errors(word, G, count)
    word_with_mistake = np.where(word_with_mistake == 0, -1, word_with_mistake)  # Заменяем 0 на -1 для декодирования
    w_t = [word_with_mistake @ H_RM(1, m)]
    for i in range(2, m + 1):
        w_t.append(w_t[-1] @ H_RM(i, m))

    # Поиск исправления с наибольшей вероятностью
    maximum = w_t[0][0]
    index = -1
    for i in range(len(w_t)):
        for j in range(len(w_t[i])):
            if abs(w_t[i][j]) > abs(maximum):
                index = j
                maximum = w_t[i][j]

    counter = sum(abs(w_t[i][j]) == abs(maximum) for i in range(len(w_t)) for j in range(len(w_t[i])))
    if counter > 1:
        print("Невозможно исправить ошибку: неоднозначное декодирование.\n")
        return

    # Исправление ошибок
    corrected_word = list(map(int, list(f"{index:0{m}b}")))
    corrected_word.append(1 if maximum > 0 else 0)
    print(f"Исправленное сообщение: {np.array(corrected_word[::-1])}")

def part_one():
    """
    Часть 1: Исследование расширенного кода Голея (24, 12, 8).
    """
    print("-------------------------------\nЧасть 1")
    G, H = G_H(B)
    print(f"Порождающая матрица G:\n{G}\nПроверочная матрица H:\n{H}")

    word = np.array([i % 2 for i in range(len(G))])
    for i in range(5):  # Тестирование для 0-4 ошибок
        word_with_mistake = create_errors(word, G, i)
        correct_mistake(word, word_with_mistake, H, B, G)
        print('')

def part_two():
    """
    Часть 2: Исследование кодов Рида-Маллера RM(1, 3) и RM(1, 4).
    """
    print("-------------------------------\nЧасть 2")

    # RM(1, 3)
    m = 3
    print(f"\nПорождающая матрица для RM(1, 3):\n{G_RM(1, m)}\n")
    word = np.array([i % 2 for i in range(len(G_RM(1, m)))])
    for j in range(1, 3):  # Тестирование для 1 и 2 ошибок
        research_with_RM(word, G_RM(1, m), j, m)

    # RM(1, 4)
    m = 4
    print(f"\nПорождающая матрица для RM(1, 4):\n{G_RM(1, m)}\n")
    word = np.array([i % 2 for i in range(len(G_RM(1, m)))])
    for j in range(1, 5):  # Тестирование для 1-4 ошибок
        research_with_RM(word, G_RM(1, m), j, m)

if __name__ == '__main__':
    part_one()
    part_two()
