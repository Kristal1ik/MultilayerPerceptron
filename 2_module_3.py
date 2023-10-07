import torch


def custom_compare(x, y):
    if str(x) != str(y):
        raise RuntimeError(f'Ожидаемое значение: {y}. Фактическое: {x}')


def to_list(x, precision=2):
    return [round(x, precision) for x in x.tolist()]


def to_list_m(m, precision=2):
    res = []

    for l in m.tolist():
        res.append([round(x, precision) for x in l])

    return res


def function_3_solution(x, y):

    return torch.transpose(torch.log(x) * (y.reshape((-1, 1)))**2, 0, 1)


def function_3_test():
    x_example_1 = torch.tensor([1, 2.71828], dtype=torch.float)
    y_example_1 = torch.tensor([2, 3, 4], dtype=torch.float)

    res_example_1 = [[0.0, 0.0, 0.0],
                     [4.0, 9.0, 16.0]]

    custom_compare(to_list_m(function_3_solution(x_example_1, y_example_1)),
                   res_example_1)

    x_example_2 = torch.tensor([72, 7.2, 2.8, 5.6], dtype=torch.float)
    y_example_2 = torch.tensor([11, 32, 4.1, -8, 1.7, 3.2, -4.9], dtype=torch.float)

    res_example_2 = [[517.48, 4379.31, 71.89, 273.71, 12.36, 43.79, 102.68],
                     [238.86, 2021.46, 33.18, 126.34, 5.71, 20.21, 47.4],
                     [124.58, 1054.33, 17.31, 65.9, 2.98, 10.54, 24.72],
                     [208.45, 1764.11, 28.96, 110.26, 4.98, 17.64, 41.36]]

    custom_compare(to_list_m(function_3_solution(x_example_2, y_example_2)),
                   res_example_2)

    print('Тест прошёл успешно!')


function_3_test()
