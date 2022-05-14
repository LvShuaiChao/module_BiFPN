import torch
import torch.nn as nn

from nets.efficientdet import BiFPN

if __name__ == '__main__':
    bifpn  = nn.Sequential(*[BiFPN(num_channels=88,
                   conv_channels=[40, 112, 320],
                   first_time=False,
                   attention=True
                   )])

    fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384]
    conv_channel_coef = {
        0: [40, 112, 320],
        1: [40, 112, 320],
        2: [48, 120, 352],
        3: [48, 136, 384],
        4: [56, 160, 448],
        5: [64, 176, 512],
        6: [72, 200, 576],
        7: [72, 200, 576],
    }
    phi = 1
    fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8]

    bifpn = nn.Sequential(
        *[BiFPN(fpn_num_filters[phi],
                conv_channel_coef[phi],
                True if _ == 0 else False,
                attention=True if phi < 6 else False)
          for _ in range(fpn_cell_repeats[phi])])

    # print(bifpn)
    p3 = torch.rand(8, 40, 80, 80)
    p4 = torch.rand(8, 112, 40, 40)
    p5 = torch.rand(8, 320, 20, 20)
    print('---------------------------------------------------')
    print('输入')
    print("p3.shape:", p3.shape)
    print("p4.shape:", p4.shape)
    print("p5.shape:", p5.shape)

    print('输出')
    features = (p3, p4, p5)
    outputs = bifpn(features)
    o3,o4,o5,_,_ = outputs
    print("o3.shape:", o3.shape)
    print("o4.shape:", o4.shape)
    print("o5.shape:", o5.shape)
