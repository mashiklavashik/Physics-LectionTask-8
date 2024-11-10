import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

m = float(input("Введите массу груза (кг): "))
k = float(input("Введите коэффициент жесткости пружины (Н/м): "))
b = float(input("Введите коэффициент сопротивления среды (Н·с/м): "))
x0 = 1.0
v0 = 0.0

def motion(t, y):
    x, v = y
    dxdt = v
    dvdt = -(b / m) * v - (k / m) * x
    return [dxdt, dvdt]

t_span = (0, 20)
t_eval = np.linspace(*t_span, 500)

sol = solve_ivp(motion, t_span, [x0, v0], t_eval=t_eval)

x = sol.y[0]
v = sol.y[1]

kinetic_energy = 0.5 * m * v**2
potential_energy = 0.5 * k * x**2
total_energy = kinetic_energy + potential_energy

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t_eval, kinetic_energy, color='b')
plt.xlabel("Время (с)")
plt.ylabel("Кинетическая энергия (Дж)")
plt.title("Кинетическая энергия от времени")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t_eval, potential_energy, color='g')
plt.xlabel("Время (с)")
plt.ylabel("Потенциальная энергия (Дж)")
plt.title("Потенциальная энергия от времени")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t_eval, total_energy, color='r')
plt.xlabel("Время (с)")
plt.ylabel("Полная механическая энергия (Дж)")
plt.title("Полная механическая энергия от времени")
plt.grid(True)

plt.tight_layout()
plt.savefig("graphics.png")
plt.show()
