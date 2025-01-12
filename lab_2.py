import numpy as np
import matplotlib.pyplot as plt

# Задаем функции и их аналитические производные
def f1(x):
    return np.sin(x)

def f1_derivative(x):
    return np.cos(x)

def f2(x):
    return np.exp(x)

def f2_derivative(x):
    return np.exp(x)

# Численные методы дифференцирования
def right_difference(y, h):
    return (y[1:] - y[:-1]) / h  # Правая разностная производная

def left_difference(y, h):
    return (y[1:] - y[:-1]) / h  # Левая разностная производная

def central_difference(y, h):
    return (y[2:] - y[:-2]) / (2 * h)  # Центральная разностная производная

def boundary_left(y, h):
    return (-3 * y[0] + 4 * y[1] - y[2]) / (2 * h)  # Производная на левой границе

def boundary_right(y, h):
    return (y[-3] - 4 * y[-2] + 3 * y[-1]) / (2 * h)  # Производная на правой границе

# Функция для вычисления производных и построения графиков
def compute_and_plot(f, f_derivative, a, b, initial_h):
    h_values = [initial_h / (2 ** i) for i in range(5)]  # Уменьшение шага в 2, 4, 8, 16 раз
    mse_results = []  # Для хранения СКО для каждого шага

    for h in h_values:
        # Сетка разбиения и значения функции
        x = np.arange(a, b + h, h)
        y = f(x)

        # Численные производные
        dydx_right = right_difference(y, h)
        dydx_left = left_difference(y, h)
        dydx_central = central_difference(y, h)

        # Обработка границ
        dydx_left_boundary = boundary_left(y, h)
        dydx_right_boundary = boundary_right(y, h)

        # Полный массив производных
        dydx = np.zeros_like(x)  # Инициализация массива
        dydx[0] = dydx_left_boundary  # Левая граница
        dydx[-1] = dydx_right_boundary  # Правая граница
        dydx[1:-1] = central_difference(y, h)  # Центральная производная

        # Истинная аналитическая производная
        true_derivative = f_derivative(x)

        # Вычисление СКО (среднеквадратичного отклонения)
        mse = np.sqrt(np.mean((dydx - true_derivative) ** 2))
        mse_results.append(mse)

        # Построение графиков
        plt.figure(figsize=(10, 6))
        plt.plot(x, true_derivative, label="Аналитическая производная", color="black", linestyle="--")
        plt.plot(x, dydx, label="Численная производная (центральная)", color="blue")
        plt.title(f"Функция: {f.__name__}, Шаг h = {h:.4f}")
        plt.xlabel("x")
        plt.ylabel("Производная")
        plt.legend()
        plt.grid()
        plt.show()

    # График зависимости СКО от шага
    plt.figure(figsize=(10, 6))
    plt.plot(h_values, mse_results, marker="o", linestyle="--", color="red")
    plt.title(f"Зависимость СКО от величины шага для {f.__name__}")
    plt.xlabel("Величина шага (h)")
    plt.ylabel("Среднеквадратичное отклонение (СКО)")
    plt.xscale("log")  # Логарифмическая шкала для шага
    plt.yscale("log")  # Логарифмическая шкала для ошибки
    plt.grid()
    plt.show()

# Параметры
a = 0  # Начало отрезка
b = 2 * np.pi  # Конец отрезка
initial_h = 0.1  # Начальный шаг

# Анализ для первой функции
print("Анализ функции f1(x) = sin(x)")
compute_and_plot(f1, f1_derivative, a, b, initial_h)

# Анализ для второй функции
print("Анализ функции f2(x) = exp(x)")
compute_and_plot(f2, f2_derivative, a, b, initial_h)