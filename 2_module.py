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


def function_1_test():
    vect_example_1 = torch.tensor([0, 1], dtype=torch.float)
    res_example_1 = -0.85

    custom_compare(round(function_1_solution(vect_example_1).item(), 2), res_example_1)

    vect_example_2 = torch.tensor([0, 3.14159], dtype=torch.float)
    res_example_2 = -0.91

    custom_compare(round(function_1_solution(vect_example_2).item(), 2), res_example_2)

    vect_example_3 = torch.tensor([6, 2, 3, 1.1], dtype=torch.float)
    res_example_3 = 0.54

    custom_compare(round(function_1_solution(vect_example_3).item(), 2), res_example_3)

    print('Тест прошёл успешно!')


def function_1_solution(x):
    return torch.sum((torch.sin(x) - torch.cos(x)) / (x ** 2 + 1))




function_1_test()


