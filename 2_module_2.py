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


def function_2_solution(x, y):

    return torch.cat((x/torch.sum(y), y/torch.sum(x)))


def function_2_test():
    x_example_1 = torch.tensor([1, 2, 3], dtype=torch.float)
    y_example_1 = torch.tensor([1, 1, 1], dtype=torch.float)

    res_example_1 = [0.33, 0.67, 1.0, 0.17, 0.17, 0.17]

    custom_compare(to_list(function_2_solution(x_example_1, y_example_1)),
                   res_example_1)

    x_example_2 = torch.tensor([2, 1, 9, 34], dtype=torch.float)
    y_example_2 = torch.tensor([22, 17, -1], dtype=torch.float)

    res_example_2 = [0.05, 0.03, 0.24, 0.89, 0.48, 0.37, -0.02]

    custom_compare(to_list(function_2_solution(x_example_2, y_example_2)),
                   res_example_2)

    print('Тест прошёл успешно!')


function_2_test()
