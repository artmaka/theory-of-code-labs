import numpy as np
import random

# 1.1 Кодирование сообщения
def encode_message(u, g):
    """
    Кодирует сообщение с использованием порождающего многочлена.

    Args:
        u (np.ndarray): Исходное сообщение (вектор).
        g (np.ndarray): Порождающий многочлен.

    Returns:
        np.ndarray: Закодированное сообщение.
    """
    return np.polymul(u, g) % 2

# 1.2 Генерация ошибок
def introduce_mistakes(word, error_count):
    """
    Вносит заданное количество ошибок в сообщение.

    Args:
        word (np.ndarray): Вектор сообщения.
        error_count (int): Количество ошибок.

    Returns:
        np.ndarray: Сообщение с внесенными ошибками.
    """
    n = len(word)
    error_positions = random.sample(range(n), error_count)
    print(f"Позиции ошибок: {error_positions}")
    for pos in error_positions:
        word[pos] ^= 1  # Инвертируем бит
    return word

# 1.3 Генерация пакета ошибок
def introduce_pack_mistakes(word, t):
    """
    Вносит пакет из t последовательных ошибок в сообщение.

    Args:
        word (np.ndarray): Вектор сообщения.
        t (int): Длина пакета ошибок.

    Returns:
        np.ndarray: Сообщение с внесенными пакетными ошибками.
    """
    n = len(word)
    start_pos = random.randint(0, n - t)
    for i in range(t):
        word[(start_pos + i) % n] ^= 1  # Инвертируем биты пакета
    print(f"Пакет ошибок внесён в позиции от {start_pos} до {(start_pos + t - 1) % n}")
    return word

# 2.1 Проверка ошибки
def is_this_error(error, t):
    """
    Проверяет, можно ли исправить ошибку заданной длины t.

    Args:
        error (np.ndarray): Вектор ошибки.
        t (int): Максимальная длина ошибки.

    Returns:
        bool: True, если ошибка допустимой длины, иначе False.
    """
    error = np.trim_zeros(error, 'f')
    error = np.trim_zeros(error, 'b')
    return len(error) <= t and len(error) != 0

# 2.2 Декодирование сообщения
def decode_message(w, g, t, is_packet):
    """
    Декодирует сообщение с использованием синдрома и порождающего многочлена.

    Args:
        w (np.ndarray): Сообщение с ошибками.
        g (np.ndarray): Порождающий многочлен.
        t (int): Максимальное количество ошибок для исправления.
        is_packet (bool): Флаг, указывающий на использование пакетных ошибок.

    Returns:
        np.ndarray: Декодированное сообщение или None, если декодирование не удалось.
    """
    n = len(w)
    s = np.polydiv(w, g)[1] % 2  # Остаток (синдром)

    for i in range(n):
        e_x = np.zeros(n, dtype=int)
        e_x[n - i - 1] = 1
        mult = np.polymul(s, e_x) % 2

        s_i = np.polydiv(mult, g)[1] % 2

        if is_packet:
            if is_this_error(s_i, t):
                e_i = np.zeros(n, dtype=int)
                e_i[i - 1] = 1
                e_x = np.polymul(e_i, s_i) % 2
                corrected = np.polyadd(e_x, w) % 2
                result = np.array(np.polydiv(corrected, g)[0] % 2).astype(int)
                return result
        else:
            if sum(s_i) <= t:
                e_i = np.zeros(n, dtype=int)
                e_i[i - 1] = 1
                e_x = np.polymul(e_i, s_i) % 2
                corrected = np.polyadd(e_x, w) % 2
                result = np.array(np.polydiv(corrected, g)[0] % 2).astype(int)
                return result
    return None

# 3.1 Исследование кода (7,4)
def investigate_code_7_4():
    """
    Выполняет исследование кодирования и декодирования для кода (7,4).

    Исследует код с добавлением случайных ошибок и их исправлением.
    """
    print("-------------------------------\nИсследование кода (7,4)\n")
    g = np.array([1, 1, 0, 1])  # Порождающий многочлен
    t = 1

    for error_count in range(1, 4):
        word = np.array([1, 0, 1, 0])
        print(f"Исходное сообщение: {word}")
        codeword = encode_message(word, g)
        print(f"Закодированное сообщение: {codeword}")
        codeword_with_mistakes = introduce_mistakes(codeword.copy(), error_count)
        print(f"Сообщение с ошибками: {codeword_with_mistakes}")
        decoded = decode_message(codeword_with_mistakes, g, t, is_packet=False)
        print(f"Декодированное сообщение: {decoded}")
        if np.array_equal(word, decoded):
            print("Исходное сообщение и декодированное совпадают.\n")
        else:
            print("Исходное сообщение и декодированное не совпадают.\n")

# 3.2 Исследование кода (15,9)
def investigate_code_15_9():
    """
    Выполняет исследование кодирования и декодирования для кода (15,9).

    Исследует код с добавлением пакетных ошибок и их исправлением.
    """
    print("-------------------------------\nИсследование кода (15,9)\n")
    g = np.array([1, 0, 0, 1, 1, 1, 1])  # Порождающий многочлен
    t = 3

    for packet_length in range(1, 5):
        word = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0])
        print(f"Исходное сообщение: {word}")
        codeword = encode_message(word, g)
        print(f"Закодированное сообщение: {codeword}")
        codeword_with_pack_mistakes = introduce_pack_mistakes(codeword.copy(), packet_length)
        print(f"Сообщение с пакетом ошибок: {codeword_with_pack_mistakes}")
        decoded = decode_message(codeword_with_pack_mistakes, g, t, is_packet=True)
        print(f"Декодированное сообщение: {decoded}")
        if np.array_equal(word, decoded):
            print("Исходное сообщение и декодированное совпадают.\n")
        else:
            print("Исходное сообщение и декодированное не совпадают.\n")


# Основная часть программы
if __name__ == '__main__':
    investigate_code_7_4()
    investigate_code_15_9()
