import torch
#
# # 8
# a1_b1 = torch.tensor([0, -1, 2, 3, 3, 1, 1, -1])
# x = torch.sum(a1_b1) / len(a1_b1)
# s = float(torch.sqrt(torch.sum((a1_b1 - x) ** 2) / len(a1_b1)))
# # print(s)
#
# 9
a1 = torch.tensor([0, -1, 2, 3])
b1 = torch.tensor([3, 1, 1, -1])

x_a = torch.sum(a1) / len(a1)
x_b = torch.sum(b1) / len(b1)

s_a = torch.sqrt(torch.sum((a1 - x_a) ** 2) / len(a1))

s_b = torch.sqrt(torch.sum((b1 - x_b) ** 2) / len(a1))

#print(x_a, x_b)
#print(s_a, s_b)

norm_a = [(i - x_a) / s_a for i in a1]
norm_b = [(i - x_b) / s_b for i in b1]

print(norm_a)
print(norm_b)
#
# # 10
#
# # a2_b2 = torch.tensor([-5, 1, 0, 2, 6, -2, -1, -1])
# # x = torch.sum(a2_b2) / len(a2_b2)
# # s2 = float(torch.sqrt(torch.sum((a2_b2 - x) ** 2) / len(a2_b2)))
# # print(s2)


# import torch
#
# t = torch.tensor([-5., 1., 0., 2.])
# t1 = torch.tensor([6., -2., -1., -1.])
#
# mean, std, var = torch.mean(t), torch.std(t), torch.var(t)
# mean1, std1, var1 = torch.mean(t1), torch.std(t1), torch.var(t1)
#
#
# t = (t - mean) / std
# t1 = (t1 - mean1) / std1
#
# print(t)
# print(t1)
#
# from torch.nn.functional import normalize # define a torch tensor
# t = torch.tensor([-5., 1., 0., 2.])
# t_2 = torch.tensor([6., -2., -1., -1.])
#
#
# t1 = normalize(t, p=1.0, dim = 0)
# t_22 = normalize(t_2, p=1.0, dim = 0)
#
#
# # print normalized tensor
# print("Normalized tensor with p=1:",
#  t1)
# print(t_22)
