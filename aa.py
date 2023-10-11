conv_layers = [

    # 1 C1
    Conv2d(
        in_channels=1,  # Число каналов на входе
        out_channels=6,  # Число каналов на выходе
        kernel_size=5,  # Размер ядра
        padding='same',  # Размер паддинга (1 элемент добавляется с каждой стороны)
        padding_mode="zeros",  # Указываем что в паддинге проставляем нули вдоль границ входного тензора
        stride=1,  # Stride - 1 (смотрим на каждую позицию)
        dilation=1,  # Dilation - 1 (ядро без пропусков прикладывается к куску изображения)
    ),
    # 2 C1_activation
    ReLU(
        Conv2d(
            in_channels=1,  # Число каналов на входе
            out_channels=6,  # Число каналов на выходе
            kernel_size=5,  # Размер ядра
            padding='same',  # Размер паддинга (1 элемент добавляется с каждой стороны)
            padding_mode="zeros",  # Указываем что в паддинге проставляем нули вдоль границ входного тензора
            stride=1,  # Stride - 1 (смотрим на каждую позицию)
            dilation=1,  # Dilation - 1 (ядро без пропусков прикладывается к куску изображения)
        )
    ),
    # 3 MP1
    MaxPool2d(kernel_size=2),

    # 4 C2
    Conv2d(
        in_channels=6,  # Число каналов на входе
        out_channels=16,  # Число каналов на выходе
        kernel_size=5,  # Размер ядра
        padding=0,  # Размер паддинга (1 элемент добавляется с каждой стороны)
        padding_mode="zeros",  # Указываем что в паддинге проставляем нули вдоль границ входного тензора
        stride=1,  # Stride - 1 (смотрим на каждую позицию)
        dilation=1,  # Dilation - 1 (ядро без пропусков прикладывается к куску изображения)
    ),
    ReLU(
        Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            padding=0,
            padding_mode="zeros",
            stride=1,
            dilation=1,
        )
    ),

    # 6 MP2
    MaxPool2d(kernel_size=2),

    # 7 C3
    Conv2d(
        in_channels=16,
        out_channels=120,
        kernel_size=5,
        padding=0,
        padding_mode="zeros",
        stride=1,
        dilation=1,
    ),

    ReLU(Conv2d(
        in_channels=16,
        out_channels=120,
        kernel_size=5,
        padding=0,
        padding_mode="zeros",
        stride=1,
        dilation=1,
    ))

]

linear_layers = [

    Linear(
        in_features=120,
        out_features=84,
    ),

    ReLU(),

    Linear(
        in_features=84,
        out_features=10,
    ),

    Softmax(dim=-1),
]

layers = conv_layers + [Flatten()] + linear_layers

le_net = Sequential(
    *layers
)
