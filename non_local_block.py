import torch
import torch.nn as nn
import torch.nn.functional as F


class NonLocalBlock(nn.Module):
    def __init__(self, in_channel, p=1, k=1, temperature=1e-3, device='cuda'):
        super(NonLocalBlock, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=p, stride=p)
        self.topk = k
        self.block_size = p
        self.softmax = nn.Softmax(dim=-1)
        self.in_channel = in_channel
        self.temperature = temperature
        self.device = device
        self.d2s = nn.PixelShuffle(p)

    def batched_cdist_l2(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.baddbmm(
            x2_norm.transpose(-2, -1),
            x1,
            x2.transpose(-2, -1),
            alpha=-2
        ).add_(x1_norm).clamp_min_(1e-30).sqrt_()
        return res

    def forward(self, target, ref, ref_align):
        # b, c, H, W
        b, c, H, W = target.shape
        p_height = target.shape[2] // self.block_size  # h
        p_width = target.shape[3] // self.block_size  # w
        N = p_height * p_width

        # b, c, hw
        pool_target = self.avgpool(target).flatten(start_dim=2, end_dim=3)
        pool_ref = self.avgpool(ref).flatten(start_dim=2, end_dim=3)

        # b, hw, hw
        D_p = self.batched_cdist_l2(pool_target.transpose(-2, -1).contiguous(), pool_ref.transpose(-2, -1).contiguous())

        # # b, hw, k
        topk_value, topk_indice = torch.topk(D_p, self.topk, dim=1, largest=False)
        topk_value = topk_value.transpose(1, 2)
        topk_indice = topk_indice.transpose(1, 2)

        # b, hw, c, p^2
        F_p_ref_align = ref_align.unfold(1, self.in_channel, self.in_channel
                                         ).unfold(2, self.block_size, self.block_size
                                                  ).unfold(3, self.block_size, self.block_size)
        F_p_ref_align = F_p_ref_align.reshape(b, -1, self.in_channel, self.block_size * self.block_size)
        F_p_ref = ref.unfold(1, self.in_channel, self.in_channel
                                         ).unfold(2, self.block_size, self.block_size
                                                  ).unfold(3, self.block_size, self.block_size)
        F_p_ref = F_p_ref.reshape(b, -1, self.in_channel, self.block_size * self.block_size)
        F_p_target = target.unfold(1, self.in_channel, self.in_channel
                                   ).unfold(2, self.block_size, self.block_size
                                            ).unfold(3, self.block_size, self.block_size)
        F_p_target = F_p_target.reshape(b, -1, self.in_channel, self.block_size * self.block_size)


        # b, hw, c, k, p^2 -> b, hw, c, kp^2
        topk_block_align = F_p_ref_align[torch.arange(b).unsqueeze(1).unsqueeze(2), topk_indice].transpose(2, 3).reshape(
            b, N, self.in_channel, -1)
        topk_block = F_p_ref[torch.arange(b).unsqueeze(1).unsqueeze(2), topk_indice].transpose(2, 3).reshape(
            b, N, self.in_channel, -1)

        # b, hw, c, p^2, kp^2
        topk_block_tile = torch.repeat_interleave(topk_block.unsqueeze(-2), self.block_size * self.block_size, -2)

        # b, hw, c, p^2, 1
        F_p_target_tile = F_p_target.unsqueeze(-1)

        # b, hw, p^2, kp^2
        d_p = torch.sum((topk_block_tile - F_p_target_tile).pow(2), dim=2).clamp_min_(1e-30).sqrt_()
        s = self.softmax(-d_p / self.temperature)

        # b, hw, c, p^2
        output = torch.matmul(topk_block_align, s.transpose(-2, -1))
        # b, cp^2, h, w
        output = output.reshape(b, N, -1).transpose(-2, -1).reshape(b, -1, p_height, p_width)
        # b, c, H, W
        output = self.d2s(output)

        return output


class NonLocal(nn.Module):
    def __init__(self, in_channel, inter_channels):
        super(NonLocalMean, self).__init__()
        self.in_channel = in_channel
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

        self.W = nn.Conv2d(inter_channels, in_channel, 1, 1, 0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

        self.g = nn.Conv2d(in_channel, inter_channels, 1, 1, 0)
        self.theta = nn.Conv2d(in_channel, inter_channels, 1, 1, 0)
        self.phi = nn.Conv2d(in_channel, inter_channels, 1, 1, 0)

    def forward(self, target, ref, ref_align):

        b, c, h, w = target.shape
        N = h * w
        # b, c, h/2*w/2
        phi = self.pool(self.phi(ref)).flatten(start_dim=2, end_dim=3)
        g = self.pool(self.g(ref_align)).flatten(start_dim=2, end_dim=3)

        # b, c, hw
        theta = self.theta(target).flatten(start_dim=2, end_dim=3)

        # b, hw, hw/4
        s = torch.matmul(theta.transpose(-2, -1).contiguous(), phi) / N

        # b, hw, c
        y = torch.matmul(s, g.transpose(-2, -1))

        # b, c, hw
        y = y.transpose(-2, -1).reshape(b, -1, h, w)

        W_y = self.W(y)
        z = W_y + target

        return z



if __name__ == '__main__':
    # img1 = Variable(torch.rand(1, 3, 720, 1280))
    # img2 = Variable(torch.rand(1, 3, 720, 1280))

    from PIL import Image
    from torchvision.transforms import ToTensor, ToPILImage, RandomCrop
    import torchvision.transforms.functional as TF

    img1 = Image.open("00000001.png").convert("RGB")
    img2 = Image.open("00000002.png").convert("RGB")
    img3 = Image.open("00000003.png").convert("RGB")
    img4 = Image.open("00000004.png").convert("RGB")
    
    # i, j, h, w = RandomCrop.get_params(img1, output_size=(64, 64))
    #
    # img1 = TF.crop(img1, i, j, h, w)
    # img2 = TF.crop(img2, i, j, h, w)
    # img3 = TF.crop(img3, i, j, h, w)
    # img4 = TF.crop(img4, i, j, h, w)
    
    img1_down = torch.unsqueeze(ToTensor()(img1_down), dim=0)
    img1 = torch.unsqueeze(ToTensor()(img1), dim=0)
    img2 = torch.unsqueeze(ToTensor()(img2), dim=0)
    img3 = torch.unsqueeze(ToTensor()(img3), dim=0)
    img4 = torch.unsqueeze(ToTensor()(img4), dim=0)

    in1 = torch.cat([img1, img3], dim=0)
    in2 = torch.cat([img2, img4], dim=0)

    model1 = NonLocalBlock(3, p=1, k=1, temperature=1e-3)
    model2 = NonLocal(3, 3)
    output = model(in2, in1, in1)
    output = model(in2, in1, in1)

    for b in range(2):
        img = ToPILImage()(output[b])
        img.save('save'+str(b)+'.jpg')
