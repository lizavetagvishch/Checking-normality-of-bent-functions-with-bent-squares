import itertools
from bitarray import bitarray
import numpy as np

def bits_to_index(bits):
    """Преобразует список или кортеж битов в индекс (целое число)"""
    idx = 0
    for bit in bits:
        idx = (idx << 1) | bit
    return idx

class BoolFunc:
    _all_inputs = list(itertools.product([0, 1], repeat=8))  # Кэш всех входов (256 штук)

    def __init__(self, expression=None, truth_table=None):
        self.max_var = 8

        if expression is not None:
            self.expression = expression
            self.truth_table = self._generate_truth_table()
        elif truth_table is not None:
            self.expression = None
            self.truth_table = truth_table
        else:
            raise ValueError("Нужно передать либо expression, либо truth_table")

    def _parse_expression(self):
        # Преобразуем выражение '13+24' -> список термов [[1, 3], [2, 4]]
        return [[int(ch) for ch in term] for term in self.expression.split('+')]

    def _evaluate_term(self, term, input_bits):
        # Вычисляет значение одного терма на входе
        return all(input_bits[var - 1] for var in term)

    def _generate_truth_table(self):
        """Формируем массив значений булевой функции с использованием битов"""
        terms = self._parse_expression()
        table = bitarray(1 << self.max_var)
        table.setall(0)

        for bits in itertools.product([0, 1], repeat=self.max_var):
            value = 0
            for term in terms:
                value ^= self._evaluate_term(term, bits)
            table[bits_to_index(bits)] = value
        return table

    def to_algebraic_normal_form(self):
        def mobius_transform(f):
            n = self.max_var
            f = f.copy()
            for i in range(n):
                step = 1 << i
                for j in range(1 << n):
                    if j & step:
                        f[j] ^= f[j ^ step]
            return f

        anf_coeffs = mobius_transform(self.truth_table)
        terms = []

        for idx, coeff in enumerate(anf_coeffs):
            if coeff == 0:
                continue
            if idx == 0:
                terms.append("1")
                continue
            monomial = []
            for i in range(self.max_var):
                if (idx >> i) & 1:
                    monomial.append(str(i + 1))
            term_str = "".join(sorted(monomial))
            terms.append(term_str)

        terms.sort(key=lambda t: (len(t), t))

        return "+".join(terms) if terms else "0"

    # def walsh_coefficients(self):
    #     """Считаем коэффициенты Уолша-Адамара для массива значений функции"""
    #     #n = self.max_var
    #     #inputs = list(itertools.product([0, 1], repeat=n))
    #     inputs = BoolFunc._all_inputs
    #     for u in inputs:
    #         total = 0
    #         for x, fx in zip(inputs, self.truth_table):
    #             dot = sum(a & b for a, b in zip(u, x)) % 2
    #             total += (-1) ** (fx ^ dot)#if (fx ^ dot)==0 then +=1 else
    #         yield total
    import numpy as np

    import numpy as np

    def walsh_coefficients(self):
        """Быстрое вычисление коэффициентов Уолша-Адамара с генератором yield"""
        # Преобразуем таблицу истинности в массив из ±1 (f(x) = 0 → 1, f(x) = 1 → -1)
        f = np.array(self.truth_table, dtype=np.int32)
        f = 1 - 2 * f  # 0 → 1, 1 → -1

        # Быстрое преобразование Уолша-Адамара (in-place FWHT)
        h = f.copy()
        n = len(h)
        step = 1
        while step < n:
            for i in range(0, n, step * 2):
                for j in range(step):
                    u = h[i + j]
                    v = h[i + j + step]
                    h[i + j] = u + v
                    h[i + j + step] = u - v
            step *= 2

        # Генерируем коэффициенты через yield
        for coef in h:
            yield coef

    def is_bent(self):
        """Проверяем коэффициенты Уолша-Адамара и как только один из них не равен +-16, возвращаем False"""
        #expected_abs = 2 ** (self.max_var // 2)
        expected_abs = 16
        for coef in self.walsh_coefficients():
            # if abs(coef) != expected_abs:
            if coef != expected_abs and coef != -expected_abs:
                return False
        return True

    def add(self, other):
        """Складываем массивы значений по модулю два и получаем новую функцию"""
        if len(self.truth_table) != len(other.truth_table):
            raise ValueError("Функции должны иметь одинаковое число переменных")
        # result_table = [(a ^ b) for a, b in zip(self.truth_table, other.truth_table)]
        # return BoolFunc(truth_table=result_table)
        result_table = self.truth_table ^ other.truth_table
        return BoolFunc(truth_table=result_table)

    @classmethod
    def from_anf(cls, anf_str):

        max_var = 8
        n = 1 << max_var
        truth_table = bitarray(n)
        truth_table.setall(0)

        if anf_str.strip() == "0":
            return cls(truth_table=truth_table)

        terms = anf_str.split('+')

        for idx in range(n):
            bits = [(idx >> i) & 1 for i in range(max_var)]
            val = 0
            for term in terms:
                if term == "1":
                    val ^= 1
                    continue
                term_vars = [int(c) for c in term]
                prod = 1
                for v in term_vars:
                    prod &= bits[v - 1]
                val ^= prod
            truth_table[idx] = val

        return cls(truth_table=truth_table)

    def __str__(self):
        return ''.join(map(str, self.truth_table))


class BoolSquare(BoolFunc):
    def __init__(self, matrix):
        """matrix — numpy-массив 16x16, содержащий значения функции (0 или 1)"""
        self.max_var = 8
        if isinstance(matrix, list):
            matrix = np.array(matrix)
        if matrix.shape != (16, 16):
            raise ValueError("Матрица должна быть 16x16")
        self.matrix = matrix

    @classmethod
    def from_func(cls, f: BoolFunc):
        """Создаёт BoolSquare из BoolFunc: преобразует массив значений в 16x16 матрицу"""
        flat = np.array(f.truth_table.tolist())
        matrix = flat.reshape((16, 16))
        return cls(matrix)

    def to_func(self):
        """Возвращает BoolFunc с восстановленным массивом значений"""
        flat = self.matrix.flatten()
        return BoolFunc(truth_table=bitarray(flat.tolist()))

    def is_bent_square(self):
        """
        Проверка на бент-квадрат:
        - преобразование Уолша-Адамара по строкам
        - обратное преобразование по столбцам
        - проверка, что все значения ±1
        """

        m = np.where(self.matrix == 0, 1, -1).astype(int)

        def hadamard_step(v):
            """Быстрое преобразование Уолша-Адамара"""
            n = len(v)
            h = v.copy()
            step = 1
            while step < n:
                for i in range(0, n, step * 2):
                    for j in range(step):
                        a, b = h[i + j], h[i + j + step]
                        h[i + j] = a + b
                        h[i + j + step] = a - b
                step *= 2
            return h

        transformed = np.empty_like(m)

        for i in range(16):
            transformed[i] = hadamard_step(m[i])

        for j in range(16):
            col = hadamard_step(transformed[:, j])
            col = col // 16
            if not np.all(np.isin(col, [-1, 1])):
                return False
        return True

    def __str__(self):
        lines = []
        for row in self.matrix:
            line = ' '.join(f"{val:>2}" for val in row)
            lines.append(line)
        return "\n".join(lines)

# f = BoolFunc("123+245+346+14+26+34+35+36+45+46+78")#бент функция для проверки методов
# #f = BoolFunc("127+245+346+14+26+34+35+36+45+46+78")#не бент функция
# print(f)
# anf = f.to_algebraic_normal_form()
# print(anf)
# restored = BoolFunc.from_anf(anf)
# print(f.truth_table == restored.truth_table)
#
# print("Original:", f.truth_table)
# print("Restored:", restored.truth_table)
# print(f'f is bent:{f.is_bent()}')
# for i, (a, b) in enumerate(zip(f.truth_table, restored.truth_table)):
#     if a != b:
#         print(f"Mismatch at index {i}: {a} != {b}")
#
#
# square = BoolSquare.from_func(f)
# print("Бент-квадрат?", square.is_bent_square())
#
# restored = square.to_func()
# print("Совпадают?", f.truth_table == restored.truth_table)

import itertools


def read_functions_from_file(filename):
    """
    Читает АНФ из файла построчно и создает генератор, который "отдает"
    объекты BoolFunc по одному.

    Аргументы:
    filename (str): Имя файла с АНФ, по одной на строку.

    Возвращает:
    generator: Генератор, возвращающий объекты BoolFunc.
    """
    print(f"Открываю файл '{filename}' для чтения...")
    with open(filename, 'r') as f:
        for line in f:
            # .strip() убирает лишние пробелы и символ новой строки \n в конце
            anf_string = line.strip()

            # Пропускаем пустые строки на всякий случай
            if anf_string:
                # Создаем объект BoolFunc, передавая АНФ в конструктор
                # и "отдаем" его с помощью yield
                yield BoolFunc(expression=anf_string)


# --- Пример использования ---

# Имя файла, который мы сгенерировали на предыдущем шаге
INPUT_FILENAME = "all_function_variants.txt"

# Создаем генератор
function_generator = read_functions_from_file(INPUT_FILENAME)

print("\n--- Читаем и обрабатываем первые 5 функций из файла ---")

# itertools.islice позволяет взять только несколько первых элементов из генератора,
# не считывая весь файл
for func_object in itertools.islice(function_generator, 5):
    # Теперь func_object - это полноценный объект класса BoolFunc

    # Мы можем, например, вывести его АНФ и часть таблицы истинности для проверки
    anf = func_object.to_algebraic_normal_form()  # Метод из вашего класса
    truth_table_preview = str(func_object)[:32]  # Используем __str__ из вашего класса

    print(f"  Прочитан объект для функции: {func_object}")
    print(f"  Фрагмент таблицы истинности: {truth_table_preview}...")
    print("-" * 20)

print("\nВы можете использовать генератор в цикле for для обработки всех функций в файле.")
# Пример полного цикла (закомментирован, так как он выполнится 1,048,576 раз):
#
# for func in read_functions_from_file(INPUT_FILENAME):
#     # тут можно выполнять любые действия с объектом func
#     # например, func.is_bent()
#     pass