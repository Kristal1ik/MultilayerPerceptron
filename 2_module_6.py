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


def softmax_solution(x):
    chisl = x.exp()
    znam = torch.sum(x.exp())
    return chisl/znam


def softmax_test():
    y_example_1 = torch.tensor([1, 2, 3], dtype=torch.float)

    res_example_1 = [0.09, 0.24, 0.67]

    custom_compare(to_list(softmax_solution(y_example_1)),
                   res_example_1)

    y_example_2 = torch.tensor([-1, 20, 19, -100, 2, 0], dtype=torch.float)

    res_example_2 = [0.0, 0.73, 0.27, 0.0, 0.0, 0.0]

    custom_compare(to_list(softmax_solution(y_example_2)),
                   res_example_2)

    print('Тест прошёл успешно!')


softmax_test()
