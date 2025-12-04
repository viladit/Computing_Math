from typing import Callable


def derivative(func: Callable, x: float, dx: float = 1e-5) -> float:
  """
  Вычисляет приближение производной функции 'func' в точке 'x'
  методом центральной разности.
  derivative(f, x) ≈ (f(x+dx) - f(x-dx)) / (2 * dx).
  """
  return (func(x + dx) - func(x - dx)) / (2.0 * dx)
