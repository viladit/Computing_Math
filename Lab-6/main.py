from math import sin, cos, exp
import numpy as np
import matplotlib.pyplot as plt
import os, signal

MAX_ITERS = 20

def improved_euler_method(f, xs, y0, eps):
    ys = [y0]
    h = xs[1] - xs[0]
    for i in range(len(xs) - 1):
        x, y = xs[i], ys[i]
        # predictor
        y_pred = y + h * f(x, y)
        # corrector
        y_corr = y + (h / 2) * (f(x, y) + f(x + h, y_pred))
        ys.append(y_corr)
    return ys

def fourth_order_runge_kutta_method(f, xs, y0, eps):
    ys = [y0]
    h = xs[1] - xs[0]
    for i in range(len(xs) - 1):
        x, y = xs[i], ys[i]
        k1 = h * f(x,        y)
        k2 = h * f(x + h/2,  y + k1/2)
        k3 = h * f(x + h/2,  y + k2/2)
        k4 = h * f(x + h,    y + k3)
        ys.append(y + (k1 + 2*k2 + 2*k3 + k4)/6)
    return ys


def milne_method(f, xs, y0, eps):
    n = len(xs)
    h = xs[1] - xs[0]
    ys = [y0]
    # стартовые точки через RK4
    for i in range(1, 4):
        x, y = xs[i-1], ys[i-1]
        k1 = h * f(x,        y)
        k2 = h * f(x + h/2,  y + k1/2)
        k3 = h * f(x + h/2,  y + k2/2)
        k4 = h * f(x + h,    y + k3)
        ys.append(y + (k1 + 2*k2 + 2*k3 + k4)/6)
    # predictor-corrector Milne
    for i in range(4, n):
        yp = ys[i-4] + (4*h/3)*(2*f(xs[i-3], ys[i-3])
                              - f(xs[i-2], ys[i-2])
                              + 2*f(xs[i-1], ys[i-1]))
        y_corr = yp
        while True:
            yc = ys[i-2] + (h/3)*(f(xs[i-2], ys[i-2])
                                 + 4*f(xs[i-1], ys[i-1])
                                 + f(xs[i],    y_corr))


            if abs(yc - y_corr) < eps:
                y_corr = yc
                break
            y_corr = yc
        ys.append(y_corr)
    return ys

def my_input(prompt):
    s = input(prompt)
    if s.lower() == 'q':
        print("! Выход.")
        os.kill(os.getpid(), signal.SIGINT)
    return s

def select_odu():
    print("ОДУ:")
    print("  1. y' = x + y")
    print("  2. y' = sin(x) - y")
    print("  3. y' = y / x")
    print("  4. y' = e^x")
    while True:
        choice = my_input("> Выберите ОДУ (1-4): ")
        if choice == '1':
            f = lambda x,y: x+y
            exact_y = lambda x,x0,y0: exp(x-x0)*(y0+x0+1) - x - 1
            return f, exact_y
        if choice == '2':
            f = lambda x,y: sin(x)-y
            exact_y = lambda x,x0,y0: ((2*exp(x0)*y0 - exp(x0)*sin(x0) + exp(x0)*cos(x0))
                                       / (2*exp(x))) + sin(x)/2 - cos(x)/2
            return f, exact_y
        if choice == '3':
            f = lambda x,y: y/x
            exact_y = lambda x,x0,y0: (x*y0)/x0
            return f, exact_y
        if choice == '4':
            f = lambda x,y: exp(x)
            exact_y = lambda x,x0,y0: y0 - exp(x0) + exp(x)
            return f, exact_y
        print("! Неверный ввод, попробуйте снова.")

def draw_exact(a, b, exact, x0, y0, dx=0.01):
    xs = np.arange(a, b + dx, dx)
    ys = [exact(x, x0, y0) for x in xs]
    plt.plot(xs, ys, 'g', label="Exact")

def solve(f, x0, xn, h0, y0, exact_y, eps):
    methods = [
        ("Euler Modified", improved_euler_method, 2),
        ("Runge–Kutta 4",  fourth_order_runge_kutta_method, 4),
        ("Milne",          milne_method, None),
    ]


    for name, method, p in methods:
        hi = h0
        h_next = hi
        iters = 0

        while True:
            xs = list(np.arange(x0, xn + 1e-12, hi))
            ys = method(f, xs, y0, eps)


            if p is None:
                err_all = [abs(exact_y(x, x0, y0) - y) for x, y in zip(xs, ys)]
                err = max(abs(exact_y(x, x0, y0) - y) for x,y in zip(xs, ys))
            else:
                yh = ys[-1]
                xs2 = list(np.arange(x0, xn + 1e-12, hi/2))
                y_arr = method(f, xs2, y0, eps)
                y2 = method(f, xs2, y0, eps)[-1]
                err = abs(y2 - yh)/(2**p - 1)
                err_all = [abs(exact_y(x, x0, y0) - y) for x, y in zip(xs, ys)]

            if err <= eps or iters >= MAX_ITERS:
                break
            hi /= 2
            h_next = hi
            iters += 1

        # вывод таблички для этого метода

        print(f"\n{name}: (h = {hi:.6f}, steps = {len(xs)-1}, error = {err:.2e})")
        print(f"{'   x':>8} {'y_true':>12} {'y_approx':>12}")
        print('-' * 45)
        for x, y in zip(xs, ys):
            y_t = exact_y(x, x0, y0)

            print(f"{x:8.5f} {y_t:12.6f} {y:12.6f}")

        if name == "Milne":
            print(f"\n{name}: (h = {hi:.6f}, steps = {len(xs) - 1}, error = {err:.2e})")
            print(f"{'   x':>8} {'y_true':>12} {'y_approx':>12} {'|y_true-y_miln|':>12}")
            print('-' * 45)
            for x, y, all in zip(xs, ys, err_all):
                y_t = exact_y(x, x0, y0)

                print(f"{x:8.5f} {y_t:12.6f} {y:12.6f} {all:12.20f}")


        if name == "Euler Modified":
            print("Дублированная таблица \n")
            print(f"\n{name}: (h = {hi / 2:.6f}, steps = {len(xs) - 1}, error = {err:.2e})")
            print(f"{'   x':>8} {'y_true':>12} {'y_approx':>12}")
            print('-' * 36)
            for x, y in zip(xs2, y_arr):
                y_t = exact_y(x, x0, y0)
                print(f"{x:8.5f} {y_t:12.6f} {y:12.6f}")


        # отдельный график для этого метода
        plt.figure()
        draw_exact(x0, xn, exact_y, x0, y0)
        plt.plot(xs, ys, 'o-', label=f"{name} (h={hi:.5f})")
        plt.title(name)
        plt.xlabel("x"); plt.ylabel("y")
        plt.legend(); plt.grid(True)
        plt.show()

    # общий график всех методов вместе
    plt.figure()
    draw_exact(x0, xn, exact_y, x0, y0)
    for name, method, _ in methods:
        xs = list(np.arange(x0, xn + 1e-12, hi))
        ys = method(f, xs, y0, eps)
        plt.plot(xs, ys, 'o-', label=name)
    plt.title("Сравнение методов")
    plt.xlabel("x"); plt.ylabel("y")
    plt.legend(); plt.grid(True)
    plt.show()

def main():
    f, exact_y = select_odu()
    x0  = float(my_input("> x0: "))
    xn  = float(my_input("> xn: "))
    h0  = float(my_input("> Шаг h: "))
    y0  = float(my_input("> y0: "))
    eps = float(my_input("> eps: "))
    solve(f, x0, xn, h0, y0, exact_y, eps)
    print("\nСпасибо!")

if __name__ == "__main__":
    main()
