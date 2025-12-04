import numpy as np


def read_from_keyboard():
    n = int(input("Введите размерность системы n: "))
    A = np.zeros((n, n), dtype=float)
    b = np.zeros(n, dtype=float)

    print("Введите коэффициенты матрицы A построчно:")
    for i in range(n):
        row = list(map(float, input(f"Строка {i + 1}: ").split()))
        if len(row) != n:
            raise ValueError("Неверное количество элементов в строке")
        A[i] = row

    b_vals = list(map(float, input("Введите вектор правых частей b: ").split()))
    if len(b_vals) != n:
        raise ValueError("Неверная длина вектора b")
    b = np.array(b_vals, dtype=float)

    return A, b


def read_from_file():
    filename = input("Введите имя файла: ").strip()
    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    n = int(lines[0])
    A = []
    for i in range(1, n + 1):
        row = list(map(float, lines[i].split()))
        if len(row) != n:
            raise ValueError("Неверное количество элементов в строке файла")
        A.append(row)

    b_vals = list(map(float, lines[n + 1].split()))
    if len(b_vals) != n:
        raise ValueError("Неверная длина вектора b в файле")

    return np.array(A, dtype=float), np.array(b_vals, dtype=float)


def gauss_solve(A, b):
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    A0 = A.copy()
    b0 = b.copy()
    n = len(b)

    # прямой ход
    for k in range(n - 1):
        if abs(A[k, k]) < 1e-12:
            raise ValueError(f"Нулевой диагональный элемент A[{k},{k}], метод Гаусса неприменим")
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]

    det = float(np.prod(np.diag(A)))

    # обратный ход
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(A[i, i]) < 1e-12:
            raise ValueError(f"Нулевой диагональный элемент A[{i},{i}] на обратном ходе")
        s = np.dot(A[i, i+1:], x[i+1:])
        x[i] = (b[i] - s) / A[i, i]

    residual = A0 @ x - b0
    return x, det, A, b, residual


def main():
    print("Лабораторная работа №1. Метод Гаусса (вариант 5)")
    print("Способ ввода данных:")
    print("1 — с клавиатуры")
    print("2 — из файла")

    mode = input("Ваш выбор: ").strip()

    if mode == "1":
        A, b = read_from_keyboard()
    elif mode == "2":
        A, b = read_from_file()
    else:
        print("Неверный выбор")
        return

    print("\nИсходная матрица A:")
    print(A)
    print("Вектор правых частей b:", b)

    try:
        x, det, A_tri, b_tri, r = gauss_solve(A, b)

        print("\nТреугольная матрица (после прямого хода):")
        print(A_tri)
        print("Преобразованный вектор b:")
        print(b_tri)

        print("\nОпределитель матрицы (по треугольной форме):")
        print(det)

        print("\nНайденный вектор неизвестных x:")
        print(x)

        print("\nВектор невязок r = A x − b:")
        print(r)

        print("\nСравнение с результатами библиотеки numpy:")
        try:
            x_np = np.linalg.solve(A, b)
            det_np = np.linalg.det(A)
            print("numpy x:", x_np)
            print("numpy det:", det_np)
        except Exception as e:
            print("numpy не смогла решить систему:", e)

    except ValueError as e:
        print("\nОшибка:", e)


if __name__ == "__main__":
    main()
