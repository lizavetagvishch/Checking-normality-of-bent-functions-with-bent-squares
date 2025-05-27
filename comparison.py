import numpy as np
import itertools
from pathlib import Path
from tqdm import tqdm
from bitarray import bitarray
import random
import matplotlib.pyplot as plt

from classes import BoolFunc


# def generate_forbidden_set(): #вычисляем все возможные комбинации значений для подматриц, при которых получим +-16
#     vals = range(-16, 17, 1)
#     solutions = set()
#     for a, b, g, d in itertools.product(vals, repeat=4):
#         if abs(a + b + g - d) == 32 or abs(a + b - g + d) == 32 or \
#                 abs(a - b + g + d) == 32 or abs(-a + b + g + d) == 32:
#             solutions.add((a, b, g, d))
#     return solutions


def calculate_walsh_hadamard_16(truth_table_16_bits):
    f = np.array(list(map(int, truth_table_16_bits)), dtype=np.int32)
    f = 1 - 2 * f
    h = f.copy()
    n = 16
    step = 1
    while step < n:
        for i in range(0, n, step * 2):
            for j in range(step):
                u, v = h[i + j], h[i + j + step]
                h[i + j], h[i + j + step] = u + v, u - v
        step *= 2
    return h


# def check_matrix_criteria(matrix, forbidden_set):
#     num_rows = matrix.shape[0]
#     if np.any((matrix == 16) | (matrix == -16)): return False
#     if np.any(np.sum(matrix ** 2, axis=0) > 256): return False
#     for row_indices in itertools.combinations(range(num_rows), 2):
#         for col_indices in itertools.combinations(range(16), 2):
#             sub_m = matrix[np.ix_(list(row_indices), list(col_indices))]
#             if (sub_m[0, 0], sub_m[0, 1], sub_m[1, 0], sub_m[1, 1]) in forbidden_set:
#                 return False
#     return True

# def check_matrix_criteria(matrix, forbidden_set):
#     flat = matrix.flatten()
#
#     # Условие 1: нет ±16
#     if np.any((flat == 16) | (flat == -16)):
#         return False
#
#     # Условие 2: сумма квадратов по столбцам ≤ 256
#     if np.any(np.sum(matrix ** 2, axis=0) > 256):
#         return False
#
#     # Условие 3: запрещённые четверки
#     for comb in itertools.combinations(flat, 4):
#         if comb in forbidden_set:
#             return False
#
#     return True


def transform_c1(input_matrix):
    if not isinstance(input_matrix, np.ndarray) or input_matrix.shape != (2, 2):
        raise ValueError("Входные данные для transform_c1 должны быть матрицей NumPy 2x2.")
    alpha, beta = input_matrix[0, 0], input_matrix[0, 1]
    gamma, delta = input_matrix[1, 0], input_matrix[1, 1]
    transformed_matrix = np.array([[alpha + beta + gamma - delta, alpha + beta - gamma + delta],
                                   [alpha - beta + gamma + delta, -alpha + beta + gamma + delta]])
    return 0.5 * transformed_matrix


def check_matrix_criteria(matrix):
    num_rows = matrix.shape[0]

    # Критерий 1: Проверка на наличие значений +-16 во всей матрице
    if np.any((matrix == 16) | (matrix == -16)):
        return False

    # Критерий 2: Сумма квадратов по столбцам
    if np.any(np.sum(matrix ** 2, axis=0) > 256):
        return False

    # Критерий 3: Проверка всех возможных подматриц 2x2 с прямым преобразованием C1
    for row_indices in itertools.combinations(range(num_rows), 2):
        for col_indices in itertools.combinations(range(16), 2):
            sub_m = matrix[np.ix_(list(row_indices), list(col_indices))]

            transformed_submatrix = transform_c1(sub_m)

            # Проверяем результат преобразования
            if np.any((transformed_submatrix == 16) | (transformed_submatrix == -16)):
                return False  # Преобразованная подматрица содержит +-16

    return True  # Все проверки пройдены


if __name__ == "__main__":
    QUARTIC_INDICES_TO_TEST = [1, 100, 267]
    SAMPLE_SIZE = 1000

    #FORBIDDEN_SET = generate_forbidden_set() #те комбинации значений, при которых можно получить +-16 после преобразования

    QUARTICS_FILE = "quartics_expressions.txt" #формы лангевина четвертой степени
    try:
        all_quartic_functions = [BoolFunc(expression=line.strip()) for line in Path(QUARTICS_FILE).open() if
                                 line.strip()]
        all_quartic_slices_64 = [q.truth_table[0:64] for q in all_quartic_functions]
        print(f"   Загружено и подготовлено {len(all_quartic_functions)} форм 4-й степени.")
    except (FileNotFoundError, IndexError) as e:
        print(f"ОШИБКА: {e}. Выполнение прервано.")
        exit()

    base_monomials_anf = [
        '456', '457', '458', '467', '468', '478', '567', '568', '678', '578',
        '45', '46', '47', '48', '56', '57', '58', '67', '68', '78'
    ]
    base_monomial_slices = [BoolFunc(expression=anf).truth_table[0:64] for anf in base_monomials_anf]
    num_base_monomials = len(base_monomial_slices)

    aggregated_rejection_rates = {2: [], 3: [], 4: []}


    for current_quartic_idx in QUARTIC_INDICES_TO_TEST:
        if current_quartic_idx >= len(all_quartic_slices_64):
            print(f"\nФорма четвертой степени с индексом {current_quartic_idx} не найдена. Пропускаем.")
            continue

        quartic_slice_64 = all_quartic_slices_64[current_quartic_idx]

        for num_rows in [2, 3, 4]:
            rejected_count = 0
            num_bits = num_rows * 16

            current_quartic_slice_for_rows = quartic_slice_64[:num_bits]
            current_base_slices_for_rows = [s[:num_bits] for s in base_monomial_slices]

            for _ in tqdm(range(SAMPLE_SIZE), desc=f"Форма четвертой степени {current_quartic_idx}, {num_rows} строки", leave=False):
                variant_slice = bitarray(num_bits)
                variant_slice.setall(0)

                #генерируем все комбинации, которыми можем дополнять формы лангевина
                random_combination_int = random.randint(0, (2 ** num_base_monomials) - 1)
                for i in range(num_base_monomials):
                    if (random_combination_int >> i) & 1:
                        variant_slice ^= current_base_slices_for_rows[i]

                combined_slice = current_quartic_slice_for_rows ^ variant_slice

                spectrum_rows = []
                for r in range(num_rows):
                    ti_slice = combined_slice[r * 16:(r + 1) * 16].to01()
                    spectrum_rows.append(calculate_walsh_hadamard_16(ti_slice))

                result_matrix = np.vstack(spectrum_rows)

                # if not check_matrix_criteria(result_matrix, FORBIDDEN_SET):
                #     rejected_count += 1

                if not check_matrix_criteria(result_matrix):
                    rejected_count += 1

            rejection_rate = (rejected_count / SAMPLE_SIZE) * 100
            aggregated_rejection_rates[num_rows].append(rejection_rate)

    # --- ИТОГОВЫЙ УСРЕДНЕННЫЙ ВЫВОД  ---
    print("\n\n===== Усредненные результаты анализа методом Монте-Карло =====")
    print(f"Проанализировано форм четвертой степени: {len(QUARTIC_INDICES_TO_TEST)}")
    print(f"Размер выборки для каждой формы четвертой степени: {SAMPLE_SIZE:,} случайных вариантов")
    print("-" * 60)
    print(f"| Количество строк | Усредненный % отбракованных вариантов |")
    print("-" * 60)

    plot_x_values = []
    plot_y_values = []

    for num_rows_key, rates_list in aggregated_rejection_rates.items():
        if rates_list:
            average_rate = sum(rates_list) / len(rates_list)
            print(f"|        {num_rows_key}         |                {average_rate:.4f}%               |")
            plot_x_values.append(num_rows_key)
            plot_y_values.append(average_rate)
        else:
            print(f"|        {num_rows_key}         |           Нет данных              |")
    print("-" * 60)

    print("\nАнализ всех выбранных форм четвертой степени завершен.")

    # --- Построение графика ---
    if plot_x_values and plot_y_values:

        sorted_data = sorted(zip(plot_x_values, plot_y_values))
        x_ax = [item[0] for item in sorted_data]
        y_ax = [item[1] for item in sorted_data]

        plt.figure(figsize=(8, 6))
        plt.plot(x_ax, y_ax, marker='o', linestyle='-', color='b')

        plt.title('Зависимость % отбраковки от количества строк')
        plt.xlabel('Количество анализируемых строк спектра')
        plt.ylabel('Усредненный % отбракованных вариантов')

        plt.xticks(x_ax)
        plt.grid(True, linestyle='--', alpha=0.7)

        for i, j in zip(x_ax, y_ax):
            plt.text(i, j + 0.5, f'{j:.2f}%', ha='center')

        plt.tight_layout()
        plt.show()
    else:
        print("\nНет данных для построения графика.")
