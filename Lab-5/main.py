import numpy as np
import matplotlib.pyplot as plt
from math import sin, sqrt, factorial
from prettytable import PrettyTable
from functools import reduce
import math

def lagrange_polynomial(dots, x, go=False):
    """ Многочлен Лагранжа """
    result = 0
    n = len(dots)
    table = PrettyTable()
    table.field_names = ['i', 'x_i', 'f(x_i)', 'L_i(x)', 'P(x)']
    
    for i in range(n):
        c1 = c2 = 1
        for j in range(n):
            if i != j:
                c1 *= x - dots[j][0]
                c2 *= dots[i][0] - dots[j][0]
        li_x = c1 / c2
        result += dots[i][1] * li_x
        table.add_row([i, f"{dots[i][0]:.3f}", f"{dots[i][1]:.3f}", f"{li_x:.3f}", f"{result:.3f}"])

    if go:
        print("Интерполяция Лагранжа")
        print(table)
    
    return result, table

def newton_divided_differences(dots, go=False):
    """ Многочлен Ньютона с разделенными разностями """
    n = len(dots)
    
    # разделенные разности
    divided_differences = [[dots[i][1] for i in range(n)]]
    
    for j in range(1, n):
        row = []
        for i in range(n - j):
            diff = (divided_differences[j - 1][i + 1] - divided_differences[j - 1][i]) / (dots[i + j][0] - dots[i][0])
            row.append(diff)
        divided_differences.append(row)
    
    if go:
        table = PrettyTable()
        headers = ["x", "f(x)"]
        for i in range(1, n):
            headers.append(f"Δ^{i} f(x)")
        table.field_names = headers

        for i in range(n):
            row = [f"{dots[i][0]:.3f}"]
            for j in range(n):
                if j < len(divided_differences) and i < len(divided_differences[j]):
                    row.append(f"{divided_differences[j][i]:.3f}")
                else:
                    row.append('')
            table.add_row(row)
        
        print("Разделенные разности")
        print(table)
    
    return divided_differences

def newton_polynomial(dots, x, go=False):
    """ Многочлен Ньютона с разделенными разностями """
    n = len(dots)
    divided_differences = newton_divided_differences(dots, go)
    result = divided_differences[0][0]
    table = [['i', 'x_i', 'f[x_i]', 'P(x)']]
    for i in range(1, n):
        term = divided_differences[i][0]
        for j in range(i):
            term *= (x - dots[j][0])
        result += term
        table.append([i, dots[i][0], divided_differences[i][0], result])
    return result, table

def gauss_polynomial(dots, x, go=False):
    # Извлекаем значения x и y из заданных точек
    xs = [dot[0] for dot in dots]
    ys = [dot[1] for dot in dots]
    n = len(xs) - 1  # Степень полинома
    alpha_ind = n // 2  # Средний индекс для начального значения

    # Вычисляем конечные разности
    finite_differences = []
    finite_differences.append(ys[:])  # Нулевые конечные разности (исходные значения y)
    
    # формируем конечные разности
    for k in range(1, n + 1):
        last = finite_differences[-1][:]
        finite_differences.append([last[i + 1] - last[i] for i in range(n - k + 1)])

    if go:
        table = PrettyTable()
        headers = ["x", "f(x)"]
        for i in range(1, n + 1):
            headers.append(f"Δ^{i} f(x)")
        table.field_names = headers

        for i in range(n + 1):
            row = [f"{xs[i]:.3f}"] if i < len(xs) else [""]
            for j in range(n + 1):
                if j < len(finite_differences) and i < len(finite_differences[j]):
                    row.append(f"{finite_differences[j][i]:.3f}")
                else:
                    row.append('')
            table.add_row(row)
        
        print("Конечные разности")
        print(table)

    # Шаг
    h = xs[1] - xs[0]
    max_terms = (n + 1) // 2
    t_values = [i // 2 if i % 2 == 0 else -(i // 2 + 1) for i in range(max_terms * 2)]

    result = ys[alpha_ind]
    # Первая интерполяционная формула Гаусса
    def Pn1(x):
        result = ys[alpha_ind]
        for k in range(1, n + 1):
            t_prod = 1
            for j in range(k):
                t_prod *= ((x - xs[alpha_ind]) / h + t_values[j])
            result += t_prod * finite_differences[k][len(finite_differences[k]) // 2] / math.factorial(k)
        return result

    # Вторая интерполяционная формула Гаусса
    def Pn2(x):
        result = ys[alpha_ind]
        for k in range(1, n + 1):
            t_prod = 1
            for j in range(k):
                t_prod *= ((x - xs[alpha_ind]) / h - t_values[j])
            result += t_prod * finite_differences[k][len(finite_differences[k]) // 2 - (1 - len(finite_differences[k]) % 2)] / math.factorial(k)
        return result

    # Выбор подходящей формулы в зависимости от положения x
    if x > xs[alpha_ind]:
        res = Pn1(x)
    else:
        res = Pn2(x)

    if go:
        # Создание таблицы промежуточных значений
        table = PrettyTable()
        table.field_names = ['k', 't_k', 'Промежуточное произведение', 'Текущий член']
        for k in range(n + 1):
            if k == 0:
                table.add_row([k, '-', '-', '-'])
            else:
                t_prod = 1
                t_prod_str = '1'
                for j in range(k):
                    t_prod *= ((x - xs[alpha_ind]) / h + t_values[j])
                    t_prod_str += f" * ((x - {xs[alpha_ind]:.3f}) / h + {t_values[j]})"
                current_term = t_prod * finite_differences[k][len(finite_differences[k]) // 2] / math.factorial(k)
                result += current_term
                table.add_row([k, t_values[k-1] if k-1 < len(t_values) else '-', t_prod_str, f"{current_term:.3f}"])

        print("Интерполяция Гаусса")
        print(table)

    return res


def check_equally_spaced(dots):
    """ Проверка на равностоящие узлы """
    x_values = [dot[0] for dot in dots]
    differences = np.diff(x_values)
    return np.allclose(differences, differences[0])


def plot(x, y, plot_x, plot_y, additional_point=None):
    fig, ax = plt.subplots()
    ax.spines['left'].set_position('zero')
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_color('gray')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.plot(1, 0, marker=">", ms=5, color='k', transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, marker="^", ms=5, color='k', transform=ax.get_xaxis_transform(), clip_on=False)
    ax.plot(x, y, 'o', label='Узлы')
    ax.plot(plot_x, plot_y, label='График многочлена')
    if additional_point is not None:
        ax.plot(additional_point[0], additional_point[1], 'ro', label='Значение аргумента')
    ax.legend()
    ax.set_title('График многочлена')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(min(x) - 1, max(x) + 1)
    ax.set_ylim(min(y) - 1, max(y) + 1)
    plt.show(block=False)

def getfunc(func_id):
    if func_id == '1':
        return lambda x: sqrt(x)
    elif func_id == '2':
        return lambda x: x ** 2
    elif func_id == '3':
        return lambda x: sin(x)
    else:
        return None

def make_dots(f, a, b, n):
    dots = []
    h = (b - a) / (n - 1)
    for i in range(n):
        dots.append((a, f(a)))
        a += h
    return dots

def getdata_input():
    data = {}
    print("\nВыберите метод интерполяции.")
    print(" 1 — Многочлен Лагранжа")
    print(" 2 — Многочлен Ньютона с разделенными разностями")
    print(" 3 — Многочлен Гаусса")
    while True:
        try:
            method_id = input("Метод решения: ")
            if method_id not in ('1', '2', '3'):
                raise AttributeError
            break
        except AttributeError:
            print("Метода нет в списке.")
    data['method_id'] = method_id

    print("\nВыберите способ ввода исходных данных.")
    print(" 1 — Набор точек")
    print(" 2 — Функция")
    while True:
        try:
            input_method_id = input("Способ: ")
            if input_method_id not in ('1', '2'):
                raise AttributeError
            break
        except AttributeError:
            print("Способа нет в списке.")

    dots = []
    if input_method_id == '1':
        print("Вводите координаты через пробел, каждая точка с новой строки.")
        print("Чтобы закончить, введите 'END'.")
        while True:
            try:
                current = input()
                if current == 'END':
                    if len(dots) < 2:
                        raise AttributeError
                    break
                x, y = map(float, current.split())
                dots.append((x, y))
            except ValueError:
                print("Введите точку повторно - координаты должны быть числами!")
            except AttributeError:
                print("Минимальное количество точек - две!")
    elif input_method_id == '2':
        print("\nВыберите функцию.")
        print(" 1 — √x")
        print(" 2 - x²")
        print(" 3 — sin(x)")
        while True:
            try:
                func_id = input("Функция: ")
                func = getfunc(func_id)
                if func is None:
                    raise AttributeError
                break
            except AttributeError:
                print("Функции нет в списке.")
        print("\nВведите границы отрезка.")
        while True:
            try:
                a, b = map(float, input("Границы отрезка: ").split())
                if a > b:
                    a, b = b, a
                break
            except ValueError:
                print("Границы отрезка должны быть числами, введенными через пробел.")
        print("\nВыберите количество узлов интерполяции.")
        while True:
            try:
                n = int(input("Количество узлов: "))
                if n < 2:
                    raise ValueError
                break
            except ValueError:
                print("Количество узлов должно быть целым числом > 1.")
        dots = make_dots(func, a, b, n)
        
    print(dots)
    data['dots'] = dots

    print("\nВведите значение аргумента для интерполирования.")
    while True:
        try:
            x = float(input("Значение аргумента: "))
            break
        except ValueError:
            print("Значение аргумента должно быть числом.")
    data['x'] = x

    if method_id in ('3') and not check_equally_spaced(dots):
        print("Предупреждение: узлы не равностоящие. Это может привести к некорректным результатам для выбранного метода.")
    elif method_id in ('2') and check_equally_spaced(dots):
        print("Предупреждение: узлы не равностоящие. Это может привести к некорректным результатам для выбранного метода.")
    
    return data

def main():
    print("\tЛабораторная работа #5 (8)")
    print("\t   Интерполяция функций")

    data = getdata_input()
    x = np.array([dot[0] for dot in data['dots']])
    y = np.array([dot[1] for dot in data['dots']])
    plot_x = np.linspace(np.min(x), np.max(x), 100)
    plot_y = None
    table = None

    if data['method_id'] == '1':
        answer, table = lagrange_polynomial(data['dots'], data['x'], True)
        plot_y = [lagrange_polynomial(data['dots'], x)[0] for x in plot_x]
    elif data['method_id'] == '2':
        answer, table = newton_polynomial(data['dots'], data['x'], True)
        plot_y = [newton_polynomial(data['dots'], x)[0] for x in plot_x]
    elif data['method_id'] == '3':
        answer = gauss_polynomial(data['dots'], data['x'], True)
        plot_y = [gauss_polynomial(data['dots'], x) for x in plot_x]
    else:
        answer = None

    if answer is not None:
        if table is not None and data['method_id'] != '1':
            print("\nТаблица промежуточных значений:")
            for row in table:
                print(row)
                
        plot(x, y, plot_x, plot_y, [data['x'], answer])
    print("\n\nРезультаты вычисления.")
    print(f"Приближенное значение функции: {answer}")

    input("\n\nНажмите Enter, чтобы выйти.")

main()
