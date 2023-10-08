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


def mae_solution(x):
    pass


def mae_general_solution(y, y_pred):
    sum_ = 0
    kk = len(y[0])
    dd = len(y)
    for k in range(dd):
        for d in range(kk):
            sum_ += torch.abs(y[k][d] - y_pred[k][d])
    return sum_ / (kk*dd)

    # return torch.sum(torch.sum(torch.abs(y-y_pred))) / (len(y) * len(y))


def mae_general_test():
    y_example_1 = torch.tensor([[1, -1], [1, 1]], dtype=torch.float)
    y_pred_example_1 = torch.tensor([[1, -1], [-1, -1]], dtype=torch.float)

    res_example_1 = 1.0

    custom_compare(round(mae_general_solution(y_example_1, y_pred_example_1).item(), 2),
                   res_example_1)

    y_example_2 = torch.tensor([[10, 22, -3], [11, 28, 0], [1, -1, 11], [22, 39, -7]], dtype=torch.float)
    y_pred_example_2 = torch.tensor([[11, 20, 5], [5, 30, -2], [1, -1, 0], [18, 30, -7]], dtype=torch.float)

    res_example_2 = 3.75

    custom_compare(round(mae_general_solution(y_example_2, y_pred_example_2).item(), 2),
                   res_example_2)

    print('Тест прошёл успешно!')


mae_general_test()
