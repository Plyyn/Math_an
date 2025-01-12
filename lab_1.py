import numpy as np
import matplotlib.pyplot as plt

def f(x):
    """Параболическая функция для минимизации."""
    return (x - 2)**2 + 1

def golden_section_search(a, b, epsilon):
    """Метод золотого сечения для нахождения минимума функции."""
    phi = (1 + np.sqrt(5)) / 2  # Золотое сечение
    resphi = 2 - phi  # 1/(phi^2)

    # Начальные точки
    yk = a + resphi * (b - a)
    zk = b - resphi * (b - a)

    # Список для хранения значений функции для графика
    points = []

    while abs(b - a) > epsilon:
        points.append((yk + zk) / 2)  # Сохраняем текущую точку минимума

        if f(yk) <= f(zk):
            b = zk
            zk = yk
            yk = a + resphi * (b - a)
        else:
            a = yk
            yk = zk
            zk = b - resphi * (b - a)

    minimum = (a + b) / 2
    return minimum, points

# Параметры
epsilon = 0.01  # Точность

# Список интервалов для тестирования
intervals = [
    (0, 4),    # Унимодальный
    (1, 3),    # Унимодальный
    (-10, 0)    # Не унимодальный
]

# Исследование каждого интервала
for a0, b0 in intervals:
    minimum, points = golden_section_search(a0, b0, epsilon)
    print(f"Интервал: [{a0}, {b0}]")
    print(f"Минимум функции достигается в x ≈ {minimum:.4f} с f(x) ≈ {f(minimum):.4f}\n")

    # Построение графика для каждого интервала
    x = np.linspace(a0 - 1, b0 + 1, 100)
    y = f(x)

    plt.plot(x, y, label='f(x) = (x - 2)^2 + 1')
    plt.axhline(y=f(minimum), color='r', linestyle='--', label='Минимум')
    plt.scatter(points, f(np.array(points)), color='orange', label='Точки итераций')
    plt.title(f'Метод золотого сечения на интервале [{a0}, {b0}]')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid()
    plt.show()