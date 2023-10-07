from math import log, ceil
def bin(data, n):
    left = 0
    right = len(data) - 1
    for i in range(ceil(log(len(data), 2))):
        middle = (left + right) // 2
        if data[middle] == n:
            return middle+1
        elif data[middle] > n:
            right = middle - 1
        else:
            left = middle + 1

    return "Элемента нет в массиве"


print(bin([1, 3, 2, 5, 8, 7], 10))
