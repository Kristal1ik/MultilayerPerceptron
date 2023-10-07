import torch


# Ряд вспомогательных функций для проверки заданий.

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


def mae_solution(y, y_pred):
    return torch.sum(torch.abs(y - y_pred)) / len(y)

def mae_test():
    y_example_1 = torch.tensor([1, -1, 1, 1], dtype=torch.float)
    y_pred_example_1 = torch.tensor([1, -1, -1, -1], dtype=torch.float)

    res_example_1 = 1.0

    custom_compare(round(mae_solution(y_example_1, y_pred_example_1).item(), 2),
                   res_example_1)

    y_example_2 = torch.tensor([10, 20, -31, 4, -5, 7, -9], dtype=torch.float)
    y_pred_example_2 = torch.tensor([3, 25, -35, 0, 10, 7, -5], dtype=torch.float)

    res_example_2 = 5.57

    custom_compare(round(mae_solution(y_example_2, y_pred_example_2).item(), 2),
                   res_example_2)

    print('Тест прошёл успешно!')


mae_test()




