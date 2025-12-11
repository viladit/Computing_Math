import math

# Функции на выбор
def func1(x): return math.sin(x)
def func2(x): return math.cos(x)
def func3(x): return x ** 2
def func4(x): return math.log(1 + x) if x > -1 else float('nan')
def func5(x): return math.exp(-x ** 2)
def func6(x): return -2 * (x ** 3) - 3 * (x ** 2) + x + 5

def select_function(num):
    return {
        1: func1,
        2: func2,
        3: func3,
        4: func4,
        5: func5,
        6: func6
    }.get(num, func1)

def left_rectangle(f, a, b, eps):
    n = 4
    p = 1
    while True:
        h = (b - a) / n
        I1 = h * sum(f(a + i * h) for i in range(n))
        h2 = h / 2
        I2 = h2 * sum(f(a + i * h2) for i in range(2 * n))
        if abs(I2 - I1) / (2 ** p - 1) < eps:
            print(f"n = {2 * n} | I = {I2}")
            return I2, 2 * n
        n *= 2

def right_rectangle(f, a, b, eps):
    n = 4
    p = 1
    while True:
        h = (b - a) / n
        I1 = h * sum(f(a + (i + 1) * h) for i in range(n))
        h2 = h / 2
        I2 = h2 * sum(f(a + (i + 1) * h2) for i in range(2 * n))
        if abs(I2 - I1) / (2 ** p - 1) < eps:
            print(f"n = {2 * n} | I = {I2}")
            return I2, 2 * n
        n *= 2

def middle_rectangle(f, a, b, eps):
    n = 4
    p = 2
    while True:
        h = (b - a) / n
        I1 = h * sum(f(a + i * h + h / 2) for i in range(n))
        h2 = h / 2
        I2 = h2 * sum(f(a + i * h2 + h2 / 2) for i in range(2 * n))
        if abs(I2 - I1) / (2 ** p - 1) < eps:
            print(f"n = {2 * n} | I = {I2}")
            return I2, 2 * n
        n *= 2

def trapezoid_method(f, a, b, eps):
    n = 4
    p = 2
    while True:
        h = (b - a) / n
        I1 = h * ((f(a) + f(b)) / 2 + sum(f(a + i * h) for i in range(1, n)))
        h2 = h / 2
        I2 = h2 * ((f(a) + f(b)) / 2 + sum(f(a + i * h2) for i in range(1, 2 * n)))
        if abs(I2 - I1) / (2 ** p - 1) < eps:
            print(f"n = {2 * n} | I = {I2}")
            return I2, 2 * n
        n *= 2

def simpson_method(f, a, b, eps):
    n = 4
    p = 4
    while True:
        # if n % 2 == 1:
        #     n += 1
        h = (b - a) / n
        I1 = f(a) + f(b)
        I1 += sum(f(a + i * h) * (4 if i % 2 else 2) for i in range(1, n))
        I1 *= h / 3

        n2 = 2 * n
        # if n2 % 2 == 1:
        #     n2 += 1
        h2 = (b - a) / n2
        I2 = f(a) + f(b)
        I2 += sum(f(a + i * h2) * (4 if i % 2 else 2) for i in range(1, n2))
        I2 *= h2 / 3

        if abs(I2 - I1) / (2 ** p - 1) < eps:
            print(f"n = {n2} | I = {I2}")
            return I2, n2
        n *= 2

def run_all_methods(f, a, b, eps):
    results = {}
    results["Левый прямоугольник"] = left_rectangle(f, a, b, eps)
    results["Правый прямоугольник"] = right_rectangle(f, a, b, eps)
    results["Средний прямоугольник"] = middle_rectangle(f, a, b, eps)
    results["Трапеция"] = trapezoid_method(f, a, b, eps)
    results["Симпсон"] = simpson_method(f, a, b, eps)
    return results

def main():
    print("Выберите функцию:")
    print("1. sin(x)\n2. cos(x)\n3. x^2\n4. ln(1+x)\n5. e^(-x^2)\n6. -2x^3-3x^2+x+5")
    func_num = int(input("Номер функции: "))
    f = select_function(func_num)

    a = float(input("Введите a: "))
    b = float(input("Введите b: "))
    eps = float(input("Введите точность (eps): "))

    print("Выберите метод:")
    print("1. Прямоугольники (левый)")
    print("2. Прямоугольники (правый)")
    print("3. Прямоугольники (средний)")
    print("4. Трапеции")
    print("5. Симпсон")
    print("6. Прямоугольники (все: левый, правый, средний)")
    method = int(input("Номер метода: "))

    if method == 1:
        left_rectangle(f, a, b, eps)
    elif method == 2:
        right_rectangle(f, a, b, eps)
    elif method == 3:
        middle_rectangle(f, a, b, eps)
    elif method == 4:
        trapezoid_method(f, a, b, eps)
    elif method == 5:
        simpson_method(f, a, b, eps)
    elif method == 6:
        print("Левых: ")
        left_rectangle(f, a, b, eps)
        print("Правых: ")
        right_rectangle(f, a, b, eps)
        print("Cредних: ")
        middle_rectangle(f, a, b, eps)


if __name__ == "__main__":
    main()
