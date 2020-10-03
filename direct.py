import numpy as np
import random
import sys
# import matplotlib.pyplot as plt
class Vector(list):

    def __sub__(self, other):
        sub = Vector([None for i in range(len(self))])
        for i in range(len(self)):
            sub[i] = self[i] - other[i]
        return sub
    def __add__(self, other):
        sub = Vector([None for i in range(len(self))])
        for i in range(len(self)):
            sub[i] = self[i] + other[i]
        return sub
    def __mul__(self, other):
        sub = [None for i in range(len(self))]
        for i in range(len(self)):
            sub[i] = self[i] * other[i]
        return sub
    def __truediv__(self, other):
        sub = [None for i in range(len(self))]
        for i in range(len(self)):
            sub[i] = self[i] / other[i]
        return sub
class Matrix(list):
    def __sub__(self, other):
        sub = [[None for i in range(len(self))] for i in range(len(self))]
        for i in range(len(self)):
            for j in range(len(self)):
                sub[i][j] = self[i][j] - other[i][j]
        return sub
    def __add__(self, other):
        add = [[None for i in range(len(self))] for i in range(len(self))]
        for i in range(len(self)):
            for j in range(len(self)):
                add[i][j] = self[i][j] + other[i][j]
        return add
    def __mult__(self, other):
        mult = [[None for i in range(len(self))] for i in range(len(self))]
        for i in range(len(self)):
            for j in range(len(self)):
                mult[i][j] = self[i][j] * other[i][j]
        return mult
    def __truediv__(self, other):
        truediv = [[None for i in range(len(self))] for i in range(len(self))]
        for i in range(len(self)):
            for j in range(len(self)):
                truediv[i][j] = self[i][j] / other[i][j]
        return truediv

def all_zero(vector):

    # print(vector)
    no_zero_counter = 0
    for item in vector:
        # print(abs(item) > 1e-9)
        if abs(item) > 1e-9:
            no_zero_counter += 1

    return no_zero_counter == 0

def zeroize(matrix, tolerance = 1e-8):
    for i in range(len(matrix[0])):

        for j in range(0, len(matrix[0])):

            if abs(matrix[i][j]) < tolerance:

                matrix[i][j] = 0

def check_dimentions(matrix): # проверка размерности матрицы

   return len(matrix) == len(matrix[0])

def norm(container, p = 2):

    _norm = 0
    if len(np.shape(container)) == 1:
        if p == 'inf':

            return max(list(map(abs, container)))

        for item in container:

            _norm += abs(item) ** p

        return _norm ** (1 / p)
    else:
        if p == 'F':
            pass
            for row in container:

                for item in row:

                    _norm += item ** 2
            return _norm ** (1 / 2)
        elif p == 'inf':

            max_sum = sum(container[0])

            for i in range(1, len(container)):
                _norm = 0
                for j in range(len(container)):

                    _norm += abs(container[i][j])

                if _norm > max_sum: max_sum =_norm

            return max_sum

        elif p == 1:

            max_sum = sum(np.array(container)[:, 0])

            for j in range(len(container)):

                _norm  = 0

                for i in range(len(container)):

                    _norm += abs(container[i][j])

                if _norm > max_sum: max_sum = _norm

            return max_sum

def dot(left_matrix, right_matrix):

    if len(np.shape(left_matrix)) == len(np.shape(right_matrix)) == 1:

        product = [0 for i in range(len(right_matrix))]

        for i in range(len(right_matrix)):

            product[i] = left_matrix[i] * right_matrix[i]

        return  product

    elif len(np.shape(left_matrix)) == 1:

        product = [0 for i in range(len(right_matrix))]

        for i in range(len(left_matrix)):

            for s in range(len(right_matrix)):

                product[i] += left_matrix[s] * right_matrix[s][i]

        return product
    elif len(np.shape(right_matrix)) == 1:

        product = [0 for i in range(len(right_matrix))]

        for i in range(len(left_matrix)):

            for s in range(len(right_matrix)):

                product[i] += left_matrix[i][s] * right_matrix[s]

        return  product
    else:

        product = [[0 for i in range(len(left_matrix))] for j in range(len(right_matrix[0]))]

        for i in range(len(left_matrix)):

             for j in range(len(right_matrix[0])):

                product[i][j] = 0

                for s in range(len(right_matrix)):

                    product[i][j] += left_matrix[i][s] * right_matrix[s][j]

        return  product
        pass
    pass

def find_general(matrix: 'матрица системы', vector: 'правая часть', k: 'номер столбца'): # поиск главного элемента

    general_element = matrix[k][k]
    general_index = k

    for i in range(k, len(matrix[0])):

        if abs(matrix[i][k]) > abs(general_element):

            general_element = matrix[i][k]
            general_index = i

    matrix[k], matrix[general_index] = matrix[general_index], matrix[k]
    vector[k], vector[general_index] = vector[general_index], vector[k]

def find_full_general(matrix, vector, order, k):

    general_element = matrix[k][k]
    general_i = k
    general_j = k
    # -----поиск главного элемента-----#
    for i in range(k, len(matrix)):

        for j in range(k, len(matrix)):

            if abs(matrix[i][j]) > abs(general_element):

                general_element = matrix[i][j]
                general_i = i #edited
                general_j = j#edited
    #----обмен k-й строки и строки, содержащей главный элемент----#
    matrix[k], matrix[general_i] = matrix[general_i], matrix[k]
    vector[k], vector[general_i] = vector[general_i], vector[k]

    #----обмен k-го столбца и столбца, содержащего главный элемент----#
    for j in range(len(matrix)):#edited

        matrix[j][k], matrix[j][general_j] = matrix[j][general_j], matrix[j][k]#edited
    #----запоминание нового порядка неизвестных----#
    order[k], order[general_j] = order[general_j], order[k]#edited
    # print('matrix: ')#edited
    # for row in matrix: print(row)#edited
    # print('vector: ', vector)#edited
    # print('order: ', order)#edited

def restore_order(solution, order):#edited
    # ----восстановление порядка сортировкой пузырьком
    for i in range(len(solution)):#edited

        for j in range(len(solution) - 1):#edited

            if order[j + 1] < order[j]:#edited

                order[j + 1], order[j] = order[j], order[j + 1]#edited

                solution[j + 1], solution[j] = solution[j], solution[j + 1]#edited

def reverse_motion(matrix, vector):      # ------Обратный ход-------#

    solution = [None for i in vector]

    for i in range(len(matrix) - 1, -1, -1):

        summ = 0

        for j in range(i + 1, len(matrix)):

            summ += matrix[i][j] * solution[j]

        solution[i] = (vector[i] - summ) / matrix[i][i]

    return solution

def gauss_method(init_matrix: 'матрица системы', init_vector: 'правая часть'):

    order = [i for i in range(len(init_vector))]

    matrix = [rows.copy() for rows in init_matrix]
    vector = init_vector.copy()

    #------Прямой ход-------#
    for k in range(len(matrix[0]) - 1):

        # find_general(matrix, vector, k)

        find_full_general(matrix, vector, order, k)

        for i in range(k + 1, len(matrix[0])):

            c = matrix[i][k] / matrix[k][k]

            for j in range(k, len(matrix[0])):

                matrix[i][j] = matrix[i][j] - c * matrix[k][j]

            vector[i] = vector[i] - c * vector[k]

        if all_zero(matrix[k]): return None, matrix, vector

    if all_zero(matrix[-1]): return None, matrix, vector

    # зануление элементов под главной диагональю
    zeroize(matrix)

    solution = reverse_motion(matrix, vector)

    restore_order(solution, order)

    return solution, matrix, vector


def qr_factor(init_matrix):

    matrix = [rows.copy() for rows in init_matrix]

    T = np.eye(len(matrix), len(matrix))

    for j in range(len(matrix)):

        for i in range(j + 1, len(matrix)):

            if abs(matrix[i][j]) < 1e-10:
                continue
            # T_cur = np.eye(len(matrix), len(matrix))

            c = matrix[j][j] / np.sqrt(matrix[j][j] ** 2 + matrix[i][j] ** 2)

            s = matrix[i][j] / np.sqrt(matrix[j][j] ** 2 + matrix[i][j] ** 2)

            for col in range(len(matrix)):
                # it_counter += 1

                temp = matrix[j][col]

                matrix[j][col] = c * temp + s * matrix[i][col]

                matrix[i][col] = -s * temp + c * matrix[i][col]

                temp = T[j][col]

                T[j][col] = c * temp + s * T[i][col]

                T[i][col] = -s * temp + c * T[i][col]

        if all_zero(matrix[j]): return None, matrix

    zeroize(matrix)

    return T, matrix

#новая версия QR разложения
def qr_decomposition(init_matrix, init_vector):

    vector = init_vector.copy()

    T, R  = qr_factor(init_matrix)

    if isinstance(T, type(None)): return None, init_matrix, init_vector

    vector = dot(T, vector)

    solution = reverse_motion(R, vector)

    return solution, R, vector

def inverse(init_matrix):

    E = list(np.eye(len(matrix), len(matrix)))

    inverted = []

    T, R = qr_factor(init_matrix)

    if isinstance(T, type(None)): return None

    for i in range(len(E)):

        inverted.append(reverse_motion(R, dot(T, E[i])))

    inverted = np.transpose(inverted)

    zeroize(inverted)

    return list(inverted)

def cond(matrix, p = 'F'):

    return norm(matrix, p) * norm(inverse(matrix), p)

def estimate(init_matrix, init_vector):

    right_matrix = [init_vector.copy() for i in init_vector]
    for i in range(len(right_matrix)):

        for j in range(len(right_matrix)):

            a = round(random.randint(-1, 1))

            right_matrix[i][j] =  right_matrix[i][j] + a * 0.01

    sol_matrix = []
    for i in range(len(right_matrix)):

        solution, _, _ = gauss_method(init_matrix, right_matrix[i])

        sol_matrix.append(solution)

    solution, _, _ = gauss_method(init_matrix, init_vector)

    estimate_vector = []

    for i in range(len(sol_matrix)):

        delta_x = norm(np.array(sol_matrix[i]) - np.array(solution)) / norm(solution)

        delta_b = norm(np.array(right_matrix[i]) - np.array(init_vector)) / norm(init_vector)

        estimate_vector.append(delta_x / delta_b)

    return max(estimate_vector)
    pass

if __name__ == '__main__':

    path = '/Users/zhursvlevy/Downloads/P_DAT1.txt'

    # np.set_printoptions(precision = 20)

    system = np.array([rows.split() for rows in open(path) if '*' not in rows])

    matrix = system[:, :-1].astype(np.float64).tolist()

    vector = system[:, -1].astype(np.float64).tolist()

    print('Исходная матрица системы')
    for row in matrix:
        print(row)

    print('Правая часть: ', vector)

    if check_dimentions(matrix) == False:

        print('Матрица не квадратная')

    elif len(matrix) < len(matrix[0]):

        print('Матрица имеет бесконечно много решений')

    else:

        solution, triangle_matrix, new_vector = gauss_method(matrix, vector)

        if not solution:

            print('Матрица вырождена')

        else:
            print('метод Гаусса: ')
            print('Треугольный вид')

            for row in triangle_matrix:

                print(row)

            print('Решение: ', solution)

            print('Норма вектора невязки: ', norm(np.array(dot(matrix, solution)) - np.array(vector)))

            solution, triangle_matrix, new_vector = qr_decomposition(matrix, vector)

            print('QR разложение: ')

            print('Решение: ', solution)

            # print('вектор невязки: ', np.array(dot(matrix, solution)) - np.array(vector))

            print('Норма вектора невязки: ', norm(np.array(dot(matrix, solution)) - np.array(vector)))

            inverted = inverse(matrix)

            print('Обратная матрица: ')
            for row in inverted:
                print(row)

            E = dot(inverted, matrix)
            zeroize(E)
            print("inv(A)*A: ")
            for row in E:
                print(row)

            print('Число обусловленности матрицы при выборе нормы ||.||_1: ', cond(matrix, 1))

            print('Число обусловленности матрицы при выборе нормы ||.||_inf: ', cond(matrix, 'inf'))

            print('Оценка снизу: ', estimate(matrix, vector))

