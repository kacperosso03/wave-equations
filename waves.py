#Kacper Boruta 272413
#Lista 4
# Równanie drgań harmonicznych tłumionych sympy
import numpy as np
import matplotlib.pyplot as plt
from sympy import Function, dsolve, Eq, symbols, init_printing
from scipy.integrate import odeint
plt.style.use('ggplot')
init_printing(use_latex=True)

t = symbols('t')
x = Function('x')
b = 0.2
w = 3
dx_dt = Eq(x(t).diff(t,2) + (2*b)*x(t).diff(t)+(w*w)*x(t),0)
dx_dt
solution = dsolve(dx_dt, ics = {x(0):0,x(t).diff(t).subs(t,0):1})
solution
t_sympy = np.linspace(0,100,500)
x_sympy = [solution.subs(t, val).rhs for val in t_sympy]
fig, ax = plt.subplots(figsize=(7,7),tight_layout=True)
ax.plot(t_sympy,x_sympy,'-')
ax.set_xlim(-1, 20)
ax.set_xlabel('$t$')
ax.set_ylabel('$x(t)$')
ax.set_title("Równanie drgań harmonicznych tłumionych metodą sympy")
plt.show()

# Równanie drgań harmonicznych scipy


def drgania(x,t_scipy, b, w):
    dx_dt = x[1]
    dv_dt = -2*b*x[1] - w*w*x[0]
    return [dx_dt, dv_dt]

b = 0.2
w = 3

t_scipy = np.linspace(0, 100, 500)
x = [0,1]
x_scipy = odeint(drgania, x, t_scipy, args=(b, w))

fig, ax = plt.subplots(figsize=(7, 7), tight_layout=True)
ax.plot(t_scipy, x_scipy[:, 0], '-')
ax.set_xlim(-1, 20)
ax.set_xlabel('$t$')
ax.set_ylabel('$x(t)$')
ax.set_title("Równanie drgań harmonicznych tłumionych metodą scipy")

plt.show()

# Wykres porównawczy
fig, ax = plt.subplots(figsize=(7, 7), tight_layout=True)
ax.plot(t_sympy, x_sympy, '-', label='sympy')
ax.plot(t_scipy, x_scipy[:, 0], '-', label='scipy')
ax.set_xlim(0, 50)
ax.set_xlabel('$t$')
ax.set_ylabel('$x(t)$')
ax.legend()

plt.show()
# blad aproksymacji
#wartosc sredniego bledu aproksymacji
def blad(a, b):
    n = len(a)
    suma = 0.0
    for i in range(n):
        suma = suma + abs(a[i] - b[i])
    return suma / n

bladx = blad(x_sympy,x_scipy[:, 0])

print("średni błąd aproksymacji x:", bladx)

def bladkw(a,b):
    n=len(a)
    mse = (1/n)*sum((a-b)**2)
    return mse
bladmse = bladkw(x_sympy,x_scipy[:, 0])
print("średni błąd kwadratowy x:", bladmse)


def blad2(a, b):
    n = len(a)
    blad = np.zeros_like(a)  # tablica błędów o takim samym rozmiarze jak tablica A
    for i in range(n):
        blad[i] = abs(a[i] - b[i])
    return blad

blad_2 = blad2(x_sympy[:len(x_scipy)], x_scipy[:, 0])

plt.plot(t_sympy[:len(x_scipy)], blad_2)
plt.xlabel("Czas")
plt.ylabel("Błąd")
plt.title("Błąd bezwzględny")
plt.show()

def blad2kw(a, b):
    n = len(a)
    blad = np.zeros_like(a)  # tablica błędów o takim samym rozmiarze jak tablica a
    for i in range(n):
        blad[i] = (1/n) * ((a[i] - b[i])**2)
    return blad

blad_2kw = blad2kw(x_sympy[:len(x_scipy)], x_scipy[:, 0])

plt.plot(t_sympy[:len(x_scipy)], blad_2kw)
plt.xlabel("Czas")
plt.ylabel("Błąd")
plt.title("Błąd kwadratowy")
plt.show()


#     Wykresy rozwiązań dla obu metod wyglądają podobnie, ale istnieje pewna różnica w szczegółach. Wykresy dla metody sympy i scipy mają podobne kształty, ale różnią się wartościami w poszczególnych punktach.

#
#     Wykres porównawczy pokazuje, że różnica między rozwiązaniami dla obu metod jest widoczna. Istnieje rozbieżność w wartościach rozwiązań dla tych samych punktów czasowych. W przypadku metody scipy, rozwiązanie ma większą amplitudę w porównaniu do metody sympy.
#
#     Obliczony średni błąd aproksymacji między rozwiązaniami wynosi około 0.0182. Błąd ten wskazuje na różnicę między rozwiązaniami dla tych samych punktów czasowych.