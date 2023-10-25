from lib.utils.snake.snake_cityscapes_utils import *

crop_scale = np.array([896, 384])
input_scale = np.array([896, 384])
scale_range = np.arange(0.4, 1.0, 0.1)


def augment(img, split, _data_rng, _eig_val, _eig_vec, mean, std, polys):
    # resize input
    height, width = img.shape[0], img.shape[1]
    center = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32) ##这里取图像的正中心
    scale = crop_scale
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    # random crop and flip augmentation
    flipped = False
    if split == 'train':
        scale = scale * np.random.uniform(0.25, 0.8) # 随机计算一个尺度
        seed = np.random.randint(0, len(polys))
        index = np.random.randint(0, len(polys[seed][0])) #从polys中的所有点随机取一个，获得其对应点的x和y，
        x, y = polys[seed][0][index]
        center[0] = x
        border = scale[0] // 2 if scale[0] < width else width - scale[0] // 2  ## 确立裁剪边界
        center[0] = np.clip(center[0], a_min=border, a_max=width-border)  ## 将center坐标控制在中间保留区域
        center[1] = y
        border = scale[1] // 2 if scale[1] < height else height - scale[1] // 2
        center[1] = np.clip(center[1], a_min=border, a_max=height-border)

        # flip augmentation
        if np.random.random() < 0.5:  ## 随机进行水平翻转
            flipped = True
            img = img[:, ::-1, :]
            center[0] = width - center[0] - 1 ## 翻转之后，center[0]变化，这里因为坐标从0-width-1，翻转之后即这样计算

    input_w, input_h = input_scale
    if split != 'train':
        center = np.array([width // 2, height // 2])
        scale = np.array([width, height])
        x = 32
        # input_w, input_h = (width + x - 1) // x * x, (height + x - 1) // x * x
        input_w, input_h = int((width / 0.5 + x - 1) // x * x), int((height / 0.5 + x - 1) // x * x)

    trans_input = data_utils.get_affine_transform(center, scale, 0, [input_w, input_h])  #从scale大小变换到与指定大小（896 384），获得其变换矩阵
    inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR) #仿射变换后，使用线性插值的方法获得真正送入网络的输入图像

    # color augmentation
    orig_img = inp.copy()
    inp = (inp.astype(np.float32) / 255.) #归一化
    if split == 'train':
        data_utils.color_aug(_data_rng, inp, _eig_val, _eig_vec)
        # data_utils.blur_aug(inp)

    # normalize the image
    inp = (inp - mean) / std    #数据域规整到正态分布
    inp = inp.transpose(2, 0, 1)

    output_h, output_w = input_h // snake_config.down_ratio, input_w // snake_config.down_ratio  ##输出也进行仿射变换
    trans_output = data_utils.get_affine_transform(center, scale, 0, [output_w, output_h])
    inp_out_hw = (input_h, input_w, output_h, output_w)

    return orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw

