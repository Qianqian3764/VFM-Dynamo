import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from einops import rearrange
from position import PositionEmbeddingSine
import torch
import math
from thop import clever_format
from thop import profile
import pynvml
import time

torch.pi = math.pi


def charbonnier(dist):
    return (dist ** 2 + 1e-6) ** 0.5 - 1e-3


def gradient(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy


def generate_pointcloud(rgbs, depths, ply_file, intrs, extrs, masks, scale=1.0):
    """
    Generate a colored point cloud in PLY format from a color and a depth image.
    Input:
    rgb_file -- filename of color image
    depth_file -- filename of depth image
    ply_file -- filename of ply file
    """
    points = []
    for rgb, depth, intr, extr, mask in zip(rgbs, depths, intrs, extrs, masks):
        # fast
        H, W = rgb.shape[1:]
        rgb = rgb.reshape(3, -1)
        depth = depth.reshape(-1)
        mask = mask.reshape(-1)
        x_grid, y_grid = np.meshgrid(range(W), range(H))
        grid_3d_pseudo = np.stack([x_grid.reshape(-1), y_grid.reshape(-1), np.ones_like(x_grid.reshape(-1))],
                                  axis=0)  # 3, HW
        grid_3d_cam = np.linalg.inv(intr[:3, :3]) @ (depth[None, :] * grid_3d_pseudo)  # 3 HW
        grid_3d_cam_pseudo = np.concatenate([grid_3d_cam, np.ones_like(grid_3d_cam[:1, :])], axis=0)  # 4, HW
        grid_3d_world = (extr @ grid_3d_cam_pseudo)[:3, :]  # 3 HW
        for i in range(grid_3d_world.shape[1]):
            if mask[i] == 0:
                continue
            points.append("%f %f %f %d %d %d 0\n" % (
            grid_3d_world[0, i], grid_3d_world[1, i], grid_3d_world[2, i], rgb[0, i], rgb[1, i], rgb[2, i]))
    file = open(ply_file, "w")
    file.write('''ply
            format ascii 1.0
            element vertex %d
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            property uchar alpha
            end_header
            %s
            ''' % (len(points), "".join(points)))
    file.close()
    # print("save ply, fx:{}, fy:{}, cx:{}, cy:{}".format(fx, fy, cx, cy))
    print("save ply")


def meshgrid(height, width, is_homogeneous=True):
    x = torch.ones(height).float().view(height, 1)
    # shape : (h,w)
    x = torch.matmul(x, torch.linspace(0, 1, width).view(1, width))

    y = torch.linspace(0, 1, height).view(height, 1)
    # shape : (h,w)
    y = torch.matmul(y, torch.ones(width).float().view(1, width))

    x = x * (width - 1)
    y = y * (height - 1)

    if is_homogeneous:
        ones = torch.ones(height, width).float()
        coords = torch.stack((x, y, ones), dim=2)  # shape: h,w,3
    else:
        coords = torch.stack((x, y), dim=2)  # shape: h,w,2

    # shape:  (h, w, 2 or 3)
    return coords


def flow_to_tgt_coords(src2tgt_flow):
    # shape: (#batch,2,h,w)
    batch_size, _, h, w = src2tgt_flow.shape

    # shape: (#batch,h,w,2)
    src2tgt_flow = src2tgt_flow.clone().permute(0, 2, 3, 1)

    # shape: (#batch,h,w,2)
    src_coords = meshgrid(h, w, False).repeat(batch_size, 1, 1, 1).cuda()

    tgt_coords = src_coords + src2tgt_flow

    normalizer = torch.tensor([(2. / w), (2. / h)]).repeat(batch_size, h, w, 1).float().cuda()
    # shape: (#batch,h,w,2)
    tgt_coords = tgt_coords * normalizer - 1

    # shape: (#batch,h,w,2)
    return tgt_coords


def flow_warp(src_img, src2tgt_flow):
    tgt_coords = flow_to_tgt_coords(src2tgt_flow)
    tgt_img = F.grid_sample(src_img, tgt_coords, padding_mode="border", align_corners=True)
    return tgt_img


def image_similarity(x, y, alpha=0.85):
    DSSIM = SSIM()
    return alpha * DSSIM(x, y) + (1 - alpha) * torch.abs(x - y)


def L2_norm(x, dim, keep_dims=True):
    curr_offset = 1e-10
    l2_norm = torch.norm(torch.abs(x) + curr_offset,
                         dim=dim, keepdim=keep_dims)
    return l2_norm


def schedule_depth_range_rigid2(prior_depth, ndepth, flow, scale_fac=None, type='inverse'):
    with torch.no_grad():
        # with torch.no_grad():
        B, _, H, W = prior_depth.shape
        depth_center = prior_depth
        flow_g = torch.sum(torch.abs(flow), dim=1).unsqueeze(1)
        flow_g = -0.02 * flow_g + 1

        scheduled_min_depth = depth_center / (1 + flow_g)

        scheduled_max_depth = depth_center * (1 + flow_g)

        if type == 'inverse':
            itv = torch.arange(0, ndepth, device=prior_depth.device, dtype=prior_depth.dtype,
                               requires_grad=False).reshape(1, -1, 1, 1).repeat(1, 1, H, W) / (ndepth - 1)  # 1 D H W
            inverse_depth_hypo = 1 / scheduled_max_depth + (1 / scheduled_min_depth - 1 / scheduled_max_depth) * itv
            depth_range = 1 / inverse_depth_hypo

        elif type == 'linear':
            itv = torch.arange(0, ndepth, device=prior_depth.device, dtype=prior_depth.dtype,
                               requires_grad=False).reshape(1, -1, 1, 1).repeat(1, 1, H, W) / (ndepth - 1)  # 1 D H W
            depth_range = scheduled_min_depth + (scheduled_max_depth - scheduled_min_depth) * itv

        elif type == 'log':
            itv = []
            for K in range(ndepth):
                K_ = torch.FloatTensor([K])
                itv.append(torch.exp(
                    torch.log(torch.FloatTensor([0.1])) + torch.log(torch.FloatTensor([1 / 0.1])) * K_ / (ndepth - 1)))
            itv = torch.FloatTensor(itv).unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(B, 1, H, W).to(
                scheduled_min_depth.device)  # B D H W
            depth_range = scheduled_min_depth + (scheduled_max_depth - scheduled_min_depth) * itv

        else:
            raise NotImplementedError

    return depth_range  # (B.D,H,W)


def random_image_mask(img, filter_size):
    '''
    :param img: [B x 3 x H x W]
    :param crop_size:
    :return:
    '''
    fh, fw = filter_size
    _, _, h, w = img.size()

    if fh == h and fw == w:
        return img, None

    x = np.random.randint(0, w - fw)
    y = np.random.randint(0, h - fh)
    filter_mask = torch.ones_like(img)  # B x 3 x H x W
    filter_mask[:, :, y:y + fh, x:x + fw] = 0.0  # B x 3 x H x W
    img = img * filter_mask  # B x 3 x H x W
    return img, filter_mask


def update_flow(flow, pix_coords, width, height):
    pix_coords = pix_coords / 2. + 0.5
    pix_coords[..., 0] *= (width - 1)
    pix_coords[..., 1] *= (height - 1)  # B H W 2
    pix_coords = pix_coords.permute(0, 3, 1, 2)  # B 2 H W
    return pix_coords + flow


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.
    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.
    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RoPE(nn.Module):
    def __init__(
        self,
        dim: int,
        theta: int = 100,
    ):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.freqs = nn.Parameter(torch.empty(2, self.dim))

    def _device_weight_init(self):
        # Create freqs in 1d
        freqs_1d = self.theta ** torch.linspace(0, -1, self.dim // 4)
        # duplicate freqs for rotation pairs of channels
        freqs_1d = torch.cat([freqs_1d, freqs_1d])
        # First half of channels do x, second half do y
        freqs_2d = torch.zeros(2, self.dim)
        freqs_2d[0, : self.dim // 2] = freqs_1d
        freqs_2d[1, -self.dim // 2 :] = freqs_1d
        # it's an angular freq here
        self.freqs.data.copy_(freqs_2d * 2 * torch.pi)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        angle = coords @ self.freqs
        return x * angle.cos() + rotate_half(x) * angle.sin()


def create_coordinate(h, w, start=0, end=1, device="cuda", dtype=torch.float32):
    # Create a grid of coordinates
    x = torch.linspace(start, end, h, device=device, dtype=dtype)
    y = torch.linspace(start, end, w, device=device, dtype=dtype)
    # Create a 2D map using meshgrid
    xx, yy = torch.meshgrid(x, y)
    # Stack the x and y coordinates to create the final map
    coord_map = torch.stack([xx, yy], axis=-1)[None, ...]
    coords = rearrange(coord_map, "b h w c -> b (h w) c", h=h, w=w)
    return coords


def split_feature(feature,
                  num_splits=2,
                  channel_last=False,
                  ):
    if channel_last:  # [B, H, W, C]
        b, h, w, c = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0

        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits

        feature = feature.view(b, num_splits, h // num_splits, num_splits, w // num_splits, c
                               ).permute(0, 1, 3, 2, 4, 5).reshape(b_new, h_new, w_new, c)  # [B*K*K, H/K, W/K, C]
    else:  # [B, C, H, W]
        b, c, h, w = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0

        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits

        feature = feature.view(b, c, num_splits, h // num_splits, num_splits, w // num_splits
                               ).permute(0, 2, 4, 1, 3, 5).reshape(b_new, c, h_new, w_new)  # [B*K*K, C, H/K, W/K]

    return feature
def merge_splits(splits,
                 num_splits=2,
                 channel_last=False,
                 ):
    if channel_last:  # [B*K*K, H/K, W/K, C]
        b, h, w, c = splits.size()
        new_b = b // num_splits // num_splits

        splits = splits.view(new_b, num_splits, num_splits, h, w, c)
        merge = splits.permute(0, 1, 3, 2, 4, 5).contiguous().view(
            new_b, num_splits * h, num_splits * w, c)  # [B, H, W, C]
    else:  # [B*K*K, C, H/K, W/K]
        b, c, h, w = splits.size()
        new_b = b // num_splits // num_splits

        splits = splits.view(new_b, num_splits, num_splits, c, h, w)
        merge = splits.permute(0, 3, 1, 4, 2, 5).contiguous().view(
            new_b, c, num_splits * h, num_splits * w)  # [B, C, H, W]

    return merge

def feature_add_position(feature0, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)

    if attn_splits > 1:  # add position in splited window
        feature0_splits = split_feature(feature0, num_splits=attn_splits)
        # feature1_splits = split_feature(feature1, num_splits=attn_splits)

        position = pos_enc(feature0_splits)

        feature0_splits = feature0_splits + position
        # feature1_splits = feature1_splits + position

        feature0 = merge_splits(feature0_splits, num_splits=attn_splits)
        # feature1 = merge_splits(feature1_splits, num_splits=attn_splits)
    else:
        position = pos_enc(feature0)

        feature0 = feature0 + position
        # feature1 = feature1 + position

    return feature0

class local_upsample_layer_v1(nn.Module):
    def __init__(self, feature_dim, scale=2):
        super(local_upsample_layer_v1, self).__init__()
        self.scale = scale
        self.upsample_mask = nn.Sequential(
            nn.Conv2d(feature_dim, 64, 3, stride=1, padding=1, dilation=1, bias=False),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, (2**scale)**2*9, 1, stride=1, padding=0, dilation=1, bias=False)
        )
        # self.q_proj = nn.Linear(144, 144)
        # self.k_proj = nn.Linear(144, 144)

        self.k_proj_1x1 = nn.Conv2d(144, 144, kernel_size=1, bias=True)
        self.q_proj_1x1 = nn.Conv2d(144, 144, kernel_size=1, bias=True)

        #self.norm = nn.GroupNorm(num_groups=9, num_channels=144, affine=False)
        #self.norm =  nn.BatchNorm2d(144)
        self.rope = RoPE(144)
        self.rope._device_weight_init()

    def forward(self, depth, feat):
        mask = self.upsample_mask(feat)

        # Apply Positional Encoding
        coords = create_coordinate(mask.shape[-2], mask.shape[-1])
        b, c, h, w = mask.shape
        mask = rearrange(mask, "b c h w -> b (h w) c")
        mask = self.rope(mask, coords)
        mask = rearrange(mask, "b (h w) c -> b c h w", h=h)
       # mask = feature_add_position(mask, 2, 144)

        # Process Queries
        # queries = self.q_proj(mask.view(b, c, -1).permute(0, 2, 1)
        #                                ).reshape(b *h*w,16, 1, 9) #[B*H*W, 16, 1,9]

       # queries = self.norm(queries).view(b, c, -1).permute(0, 2, 1).reshape(b * h * w, 16, 1,9) # [B*H*W, 16, 1,9]
        #queries = queries.view(b, c, -1).permute(0, 2, 1).reshape(b * h * w, 16, 1, 9)  # [B*H*W, 16, 1,9]

        x = self.q_proj_1x1(mask)  # (b, 144, h, w)
        #x = self.norm(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # (b, h, w, 144)
        queries = x.view(b * h * w, 16, 1, 9)  # (b*h*w, 16, 1, 9)  分成16组，每组9个通道

        # Process Keys and Values.
        keys = self.k_proj_1x1(mask)
        #keys = self.norm(keys)
        #mask
        keys = keys.view(-1, 9, h, w) #(b*16,9,h,w)
        keys_window = F.unfold(keys, kernel_size=9,
                                   padding=4)  # [b*16, 9*(3)^2),h*w] 滑动了H*W次，也就是19200次，每次滑动的窗口大小为c*K^2

        # keys_window = keys_window.view(b, 16,9, 3 ** 2, h*w).permute(
        #     0, 4, 1, 3, 2).view(b * h * w, 16,9, 3 ** 2)  # (B, HW, heads, C, KK)

        kw = keys_window.view(b, 16, 9 * 81, h * w)  # (B,16,81,HW)
        kw = kw.view(b, 16, 9, 81, h * w)  # (B,16,9,9,HW)
        kw = kw.permute(0, 4, 1, 2, 3).contiguous()  # (B,HW,16,9,9)
        keys_window = kw.view(b * h* w, 16, 9, 81)  # (BHW,16,9,9)

        scores = torch.matmul(queries, keys_window) / (9 ** 0.5)

        prob = torch.softmax(scores.squeeze(2), dim=-1)

        # keys = self.sft_key(keys, self.key_features_encoder(F.normalize(features, dim=1)))

        return local_upsample_v1(depth, prob, self.scale)  # B H2 W2

def local_upsample_v1(depth, mask, scale=2):
    if len(depth.shape) == 3:
        B, H, W = depth.shape
        depth = depth.unsqueeze(1)
    else:
        B, _, H, W = depth.shape

    mask = mask.contiguous().view(B, H, W, 4, 4, 81)  # (B,H,W,4,4,9)
    mask = mask.permute(0, 5, 3, 4, 1, 2).contiguous()  # (B, 9, 4, 4, H, W)


    up_ = F.unfold(depth, [9,9], padding=4)
    up_ = up_.view(B, 81, 1, 1, H, W)

    up_ = torch.sum(mask * up_, dim=1)  # B, 2**scale, 2**scale, H, W
    up_ = up_.permute(0, 3, 1, 4, 2).contiguous()    # B H 2**scale W 2**scale
    return up_.reshape(B, 2**scale*H, 2**scale*W)


def schedule_depth_range(disp, ndepth, scale_fac, min_depth, max_depth, type='inverse', is_depth=False):
    with torch.no_grad():
        B, _, H, W = disp.shape
        if not is_depth:
            disp_scaled = 1 / max_depth + disp * (1 / min_depth - 1 / max_depth)
            depth_center = 1 / disp_scaled
        else:
            depth_center = disp
        # 这里不要用min_tracker，而是用自己的min_depth，因为二者之间存在指数滑动差异，可能会造成震荡
        _max_depth = depth_center.reshape(B, -1).max(-1)[0][:, None, None, None].repeat(1, 1, H, W)  # B 1 H W
        _min_depth = depth_center.reshape(B, -1).min(-1)[0][:, None, None, None].repeat(1, 1, H, W)
        ori_depth_itv = (_max_depth - _min_depth) / 96.0  # FIXME: 96 is hardcoded
        scheduled_min_depth = depth_center - ori_depth_itv * scale_fac * ndepth / 2
        scheduled_max_depth = depth_center + ori_depth_itv * scale_fac * ndepth / 2
        scheduled_max_depth[scheduled_max_depth > _max_depth] = _max_depth[scheduled_max_depth > _max_depth]
        scheduled_min_depth[scheduled_min_depth < _min_depth] = _min_depth[scheduled_min_depth < _min_depth]

        if type == 'inverse':
            itv = torch.arange(0, ndepth, device=disp.device, dtype=disp.dtype, requires_grad=False).reshape(1, -1, 1,
                                                                                                             1).repeat(
                1, 1, H, W) / (ndepth - 1)  # 1 D H W
            inverse_depth_hypo = 1 / scheduled_max_depth + (1 / scheduled_min_depth - 1 / scheduled_max_depth) * itv
            depth_range = 1 / inverse_depth_hypo

        elif type == 'linear':
            itv = torch.arange(0, ndepth, device=disp.device, dtype=disp.dtype, requires_grad=False).reshape(1, -1, 1,
                                                                                                             1).repeat(
                1, 1, H, W) / (ndepth - 1)  # 1 D H W
            depth_range = scheduled_min_depth + (scheduled_max_depth - scheduled_min_depth) * itv

        elif type == 'log':
            itv = []
            for K in range(ndepth):
                K_ = torch.FloatTensor([K])
                itv.append(torch.exp(
                    torch.log(torch.FloatTensor([0.1])) + torch.log(torch.FloatTensor([1 / 0.1])) * K_ / (ndepth - 1)))
            itv = torch.FloatTensor(itv).unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(B, 1, H, W).to(
                scheduled_min_depth.device)  # B D H W
            depth_range = scheduled_min_depth + (scheduled_max_depth - scheduled_min_depth) * itv

        else:
            raise NotImplementedError

    return depth_range  # (B.D,H,W)


def schedule_depth_rangev2(prior_depth, ndepth, scale_fac, type='inverse'):
    with torch.no_grad():
        B, _, H, W = prior_depth.shape
        depth_center = prior_depth

        scheduled_min_depth = depth_center / (1 + scale_fac)
        scheduled_max_depth = depth_center * (1 + scale_fac)

        if type == 'inverse':
            itv = torch.arange(0, ndepth, device=prior_depth.device, dtype=prior_depth.dtype,
                               requires_grad=False).reshape(1, -1, 1, 1).repeat(1, 1, H, W) / (ndepth - 1)  # 1 D H W
            inverse_depth_hypo = 1 / scheduled_max_depth + (1 / scheduled_min_depth - 1 / scheduled_max_depth) * itv
            depth_range = 1 / inverse_depth_hypo

        elif type == 'linear':
            itv = torch.arange(0, ndepth, device=prior_depth.device, dtype=prior_depth.dtype,
                               requires_grad=False).reshape(1, -1, 1, 1).repeat(1, 1, H, W) / (ndepth - 1)  # 1 D H W
            depth_range = scheduled_min_depth + (scheduled_max_depth - scheduled_min_depth) * itv

        elif type == 'log':
            itv = []
            for K in range(ndepth):
                K_ = torch.FloatTensor([K])
                itv.append(torch.exp(
                    torch.log(torch.FloatTensor([0.1])) + torch.log(torch.FloatTensor([1 / 0.1])) * K_ / (ndepth - 1)))
            itv = torch.FloatTensor(itv).unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(B, 1, H, W).to(
                scheduled_min_depth.device)  # B D H W
            depth_range = scheduled_min_depth + (scheduled_max_depth - scheduled_min_depth) * itv

        else:
            raise NotImplementedError

    return depth_range  # (B.D,H,W)


def schedule_depth_range_geo(disp, ndepth, scale_fac, min_depth, max_depth, geo_mask, damper, type='inverse',
                             is_depth=False):
    with torch.no_grad():
        B, _, H, W = disp.shape
        if not is_depth:
            disp_scaled = 1 / max_depth + disp * (1 / min_depth - 1 / max_depth)
            depth_center = 1 / disp_scaled
        else:
            depth_center = disp
        # 这里不要用min_tracker，而是用自己的min_depth，因为二者之间存在指数滑动差异，可能会造成震荡
        _max_depth = depth_center.reshape(B, -1).max(-1)[0][:, None, None, None].repeat(1, 1, H, W)  # B 1 H W
        _min_depth = depth_center.reshape(B, -1).min(-1)[0][:, None, None, None].repeat(1, 1, H, W)
        ori_depth_itv = (_max_depth - _min_depth) / 96.0  # FIXME: 96 is hardcoded
        # geo
        scale_fac = scale_fac * torch.ones_like(_max_depth)  # B 1 H W
        scale_fac[geo_mask] /= damper

        scheduled_min_depth = depth_center - ori_depth_itv * scale_fac * ndepth / 2
        scheduled_max_depth = depth_center + ori_depth_itv * scale_fac * ndepth / 2
        scheduled_max_depth[scheduled_max_depth > _max_depth] = _max_depth[scheduled_max_depth > _max_depth]
        scheduled_min_depth[scheduled_min_depth < _min_depth] = _min_depth[scheduled_min_depth < _min_depth]

        if type == 'inverse':
            itv = torch.arange(0, ndepth, device=disp.device, dtype=disp.dtype, requires_grad=False).reshape(1, -1, 1,
                                                                                                             1).repeat(
                1, 1, H, W) / (ndepth - 1)  # 1 D H W
            inverse_depth_hypo = 1 / scheduled_max_depth + (1 / scheduled_min_depth - 1 / scheduled_max_depth) * itv
            depth_range = 1 / inverse_depth_hypo

        elif type == 'linear':
            itv = torch.arange(0, ndepth, device=disp.device, dtype=disp.dtype, requires_grad=False).reshape(1, -1, 1,
                                                                                                             1).repeat(
                1, 1, H, W) / (ndepth - 1)  # 1 D H W
            depth_range = scheduled_min_depth + (scheduled_max_depth - scheduled_min_depth) * itv

        elif type == 'log':
            itv = []
            for K in range(ndepth):
                K_ = torch.FloatTensor([K])
                itv.append(torch.exp(
                    torch.log(torch.FloatTensor([0.1])) + torch.log(torch.FloatTensor([1 / 0.1])) * K_ / (ndepth - 1)))
            itv = torch.FloatTensor(itv).unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(B, 1, H, W).to(
                scheduled_min_depth.device)  # B D H W
            depth_range = scheduled_min_depth + (scheduled_max_depth - scheduled_min_depth) * itv

        else:
            raise NotImplementedError

    return depth_range  # (B.D,H,W)


def schedule_depth_range_z(disp, ndepth, scale_fac, min_depth, max_depth, z_trans, type='inverse', is_depth=False):
    with torch.no_grad():
        B, _, H, W = disp.shape
        if not is_depth:
            disp_scaled = 1 / max_depth + disp * (1 / min_depth - 1 / max_depth)
            depth_center = 1 / disp_scaled
        else:
            depth_center = disp
        _max_depth = depth_center.reshape(B, -1).max(-1)[0][:, None, None, None].repeat(1, 1, H, W)  # B 1 H W
        _min_depth = depth_center.reshape(B, -1).min(-1)[0][:, None, None, None].repeat(1, 1, H, W)
        ori_depth_itv = (_max_depth - _min_depth) / 96.0  # FIXME: 96 is hardcoded
        z_trans = z_trans[:, None, None, None].repeat(1, 1, H, W)  # B 1 H W
        scheduled_min_depth = depth_center - ori_depth_itv * scale_fac * ndepth / 2 * z_trans
        scheduled_max_depth = depth_center + ori_depth_itv * scale_fac * ndepth / 2 * z_trans
        scheduled_max_depth[scheduled_max_depth > _max_depth] = _max_depth[scheduled_max_depth > _max_depth]
        scheduled_min_depth[scheduled_min_depth < _min_depth] = _min_depth[scheduled_min_depth < _min_depth]

        if type == 'inverse':
            itv = torch.arange(0, ndepth, device=disp.device, dtype=disp.dtype, requires_grad=False).reshape(1, -1, 1,
                                                                                                             1).repeat(
                1, 1, H, W) / (ndepth - 1)  # 1 D H W
            inverse_depth_hypo = 1 / scheduled_max_depth + (1 / scheduled_min_depth - 1 / scheduled_max_depth) * itv
            depth_range = 1 / inverse_depth_hypo

        elif type == 'linear':
            itv = torch.arange(0, ndepth, device=disp.device, dtype=disp.dtype, requires_grad=False).reshape(1, -1, 1,
                                                                                                             1).repeat(
                1, 1, H, W) / (ndepth - 1)  # 1 D H W
            depth_range = scheduled_min_depth + (scheduled_max_depth - scheduled_min_depth) * itv

        elif type == 'log':
            itv = []
            for K in range(ndepth):
                K_ = torch.FloatTensor([K])
                itv.append(torch.exp(
                    torch.log(torch.FloatTensor([0.1])) + torch.log(torch.FloatTensor([1 / 0.1])) * K_ / (ndepth - 1)))
            itv = torch.FloatTensor(itv).unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(B, 1, H, W).to(
                scheduled_min_depth.device)  # B D H W
            depth_range = scheduled_min_depth + (scheduled_max_depth - scheduled_min_depth) * itv

        else:
            raise NotImplementedError

    return depth_range  # (B.D,H,W)


def schedule_depth_range_zv2(prior_depth, ndepth, scale_fac, z_trans, type='inverse'):
    with torch.no_grad():
        B, _, H, W = prior_depth.shape
        depth_center = prior_depth

        scheduled_min_depth = depth_center / (1 + scale_fac * z_trans)
        scheduled_max_depth = depth_center * (1 + scale_fac * z_trans)

        if type == 'inverse':
            itv = torch.arange(0, ndepth, device=prior_depth.device, dtype=prior_depth.dtype,
                               requires_grad=False).reshape(1, -1, 1, 1).repeat(1, 1, H, W) / (ndepth - 1)  # 1 D H W
            inverse_depth_hypo = 1 / scheduled_max_depth + (1 / scheduled_min_depth - 1 / scheduled_max_depth) * itv
            depth_range = 1 / inverse_depth_hypo

        elif type == 'linear':
            itv = torch.arange(0, ndepth, device=prior_depth.device, dtype=prior_depth.dtype,
                               requires_grad=False).reshape(1, -1, 1, 1).repeat(1, 1, H, W) / (ndepth - 1)  # 1 D H W
            depth_range = scheduled_min_depth + (scheduled_max_depth - scheduled_min_depth) * itv

        elif type == 'log':
            itv = []
            for K in range(ndepth):
                K_ = torch.FloatTensor([K])
                itv.append(torch.exp(
                    torch.log(torch.FloatTensor([0.1])) + torch.log(torch.FloatTensor([1 / 0.1])) * K_ / (ndepth - 1)))
            itv = torch.FloatTensor(itv).unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(B, 1, H, W).to(
                scheduled_min_depth.device)  # B D H W
            depth_range = scheduled_min_depth + (scheduled_max_depth - scheduled_min_depth) * itv

        else:
            raise NotImplementedError

    return depth_range  # (B.D,H,W)


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)  # B 3 3
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def transformation_from_parameters_v2(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle).reshape(-1, 1, 1, 4, 4)  # B 1, 1, 4 4
    t = translation.clone()

    if invert:
        R = R.transpose(3, 4)
        t *= -1

    T = get_translation_matrix_v2(t)  # B H W 4 4

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M  # B H W 4 4


def get_translation_matrix_v2(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    B, H, W, _ = translation_vector.shape
    T = torch.zeros(B, H, W, 4, 4).to(device=translation_vector.device)
    T[..., 0, 0] = 1
    T[..., 1, 1] = 1
    T[..., 2, 2] = 1
    T[..., 3, 3] = 1
    T[..., :3, 3] = translation_vector

    return T  # B H W 4 4


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """

    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """

    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        # K： B 4 4
        # T: B H W 4 4
        if len(T.shape) == 5:
            B, H, W, _, _ = T.shape
            K = K[:, None, None]
            points = points.reshape(-1, 4, H, W, 1).permute(0, 2, 3, 1, 4)  # B H W 4 1
        P = torch.matmul(K, T)[..., :3, :]  # B H W 3 4

        cam_points = torch.matmul(P, points)  # B H W 3 1

        pix_coords = cam_points[..., :2, :] / (cam_points[..., 2:3, :] + self.eps)  # [B H W] 2 1
        if len(T.shape) == 5:
            pix_coords = pix_coords[..., 0]  # B H W 2
        else:
            pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
            pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class MVS_SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(MVS_SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.mask_pool = nn.AvgPool2d(3, 1)
        # self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y, mask):
        # print('mask: {}'.format(mask.shape))
        # print('x: {}'.format(x.shape))
        # print('y: {}'.format(y.shape))
        # x = x.permute(0, 3, 1, 2)  # [B, H, W, C] --> [B, C, H, W]
        # y = y.permute(0, 3, 1, 2)
        # mask = mask.permute(0, 3, 1, 2)

        # x = self.refl(x)
        # y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        SSIM_mask = self.mask_pool(mask.float())
        output = SSIM_mask * torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
        return output, SSIM_mask
        # return output.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def reproject_with_depth(depth_ref, intrinsics_ref, extri_ref2src, depth_src, intrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(extri_ref2src, np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.linalg.inv(extri_ref2src),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected


def generate_costvol(ref, src, K, invK, depth_priors, pose, num_depth_bins, backprojector, projector):
    cost_vols = []
    for batch_idx in range(len(ref)):
        ref_feat = ref[batch_idx:batch_idx + 1]
        source_feat = src[batch_idx:batch_idx + 1]
        source_feat = source_feat.repeat([num_depth_bins, 1, 1, 1])
        with torch.no_grad():
            _K = K[batch_idx:batch_idx + 1]
            _invK = invK[batch_idx:batch_idx + 1]
            depth_prior = depth_priors[batch_idx:batch_idx + 1]
            _lookup_poses = pose[batch_idx:batch_idx + 1, 0]
            world_points = backprojector(depth_prior, _invK)
            pix_locs = projector(world_points, _K, _lookup_poses).squeeze(1)
        warped = F.grid_sample(source_feat, pix_locs, padding_mode='zeros', mode='bilinear', align_corners=True)
        '''
        #######Du add
        # mask values landing outside the image (and near the border)
        # we want to ignore edge pixels of the lookup images and the current image
        # because of zero padding in ResNet
        # Masking of ref image border
        x_vals = (pix_locs[..., 0].detach() / 2 + 0.5) * (
                48 - 1)  # convert from (-1, 1) to pixel values
        y_vals = (pix_locs[..., 1].detach() / 2 + 0.5) * (128 - 1)

        edge_mask = (x_vals >= 2.0) * (x_vals <= 48- 2) * \
                    (y_vals >= 2.0) * (y_vals <= 128 - 2)
        edge_mask = edge_mask.float()

        # masking of current image
        current_mask = torch.zeros_like(edge_mask)
        current_mask[:, 2:-2, 2:-2] = 1.0
        edge_mask = edge_mask * current_mask
        ########
        '''
        # cost_vols.append(warped * ref_feat* edge_mask)  # D C H W
        cost_vols.append(warped * ref_feat)  # D C H W
    cost_vols = torch.stack(cost_vols, 0)  # B D C H W
    return cost_vols


def localmax(cost_prob, radius, casbin, min_depth_inverse, max_depth_inverse):
    pred_idx = torch.argmax(cost_prob, 1, keepdim=True).float()  # B 1 H W
    pred_idx_low = pred_idx - radius
    pred_idx = torch.arange(0, 2 * radius + 1, 1, device=pred_idx.device).reshape(1, 2 * radius + 1, 1, 1).float()
    pred_idx = pred_idx + pred_idx_low  # B M H W
    pred_idx = torch.clamp(pred_idx, 0, casbin - 1)
    pred_idx = pred_idx.long()
    regress_index = 0
    cost_prob_sum = 1e-6
    for i in range(2 * radius + 1):
        cost_prob_ = torch.gather(cost_prob, 1, pred_idx[:, i:i + 1])
        regress_index = regress_index + pred_idx[:, i:i + 1] * cost_prob_
        cost_prob_sum = cost_prob_sum + cost_prob_
    regress_index = regress_index.div_(cost_prob_sum)
    norm_mvs_depth = regress_index / (casbin - 1)  # B 1 H W
    depth_mvs = 1 / (min_depth_inverse + norm_mvs_depth[:, 0] * (max_depth_inverse - min_depth_inverse))  # B H W
    return depth_mvs


def reproject_with_depth(depth_ref, intrinsics_ref, extri_ref2src, depth_src, intrinsics_src, pixel_thres, depth_thres):
    with torch.no_grad():
        batch, width, height = depth_ref.shape[0], depth_ref.shape[-1], depth_ref.shape[-2]
        ## step1. project reference pixels to the source view
        # reference view x, y
        y_ref, x_ref = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))  # meshgrid different from numpy
        x_ref = x_ref.to(depth_ref.device)
        y_ref = y_ref.to(depth_ref.device)
        x_ref, y_ref = x_ref.reshape([1, -1]).repeat(batch, 1), y_ref.reshape([1, -1]).repeat(batch, 1)
        # reference 3D space
        xyz_ref = torch.inverse(intrinsics_ref) @ (
                    torch.stack((x_ref, y_ref, torch.ones_like(x_ref)), 1) * depth_ref.reshape([batch, 1, -1]))
        # source 3D space
        xyz_src = (extri_ref2src @ torch.cat((xyz_ref, torch.ones_like(x_ref).unsqueeze(1)), 1))[:, :3]  # B 3 HW
        # source view x, y
        K_xyz_src = intrinsics_src @ xyz_src
        xy_src = K_xyz_src[:, :2] / K_xyz_src[:, 2:3]  # B 2 HW

        ## step2. reproject the source view points with source view depth estimation
        # find the depth estimation of the source view
        x_src = xy_src[:, 0] / ((width - 1) / 2.) - 1
        y_src = xy_src[:, 1] / ((height - 1) / 2.) - 1
        proj_xy = torch.stack((x_src, y_src), dim=2).reshape(batch, height, width, 2)  # [B, H, W, 2]

        sampled_depth_src = F.grid_sample(depth_src, proj_xy, mode='bilinear', padding_mode='border',
                                          align_corners=True)
        # mask = sampled_depth_src > 0

        # source 3D space
        # NOTE that we should use sampled source-view depth_here to project back
        xyz_src = torch.inverse(intrinsics_src) @ (
                    torch.cat((xy_src, torch.ones_like(x_ref).unsqueeze(1)), 1) * sampled_depth_src.reshape(
                [batch, 1, -1]))
        # reference 3D space
        xyz_reprojected = (torch.inverse(extri_ref2src) @ torch.cat((xyz_src, torch.ones_like(x_ref).unsqueeze(1)), 1))[
                          :, :3]
        # source view x, y, depth
        depth_reprojected = xyz_reprojected[:, 2].reshape([batch, 1, height, width])
        K_xyz_reprojected = intrinsics_ref @ xyz_reprojected
        xy_reprojected = K_xyz_reprojected[:, :2] / K_xyz_reprojected[:, 2:3]
        x_reprojected = xy_reprojected[:, 0].reshape([batch, height, width])
        y_reprojected = xy_reprojected[:, 1].reshape([batch, height, width])

        # check |p_reproj-p_1| < 1
        dist = torch.sqrt((x_reprojected - x_ref.reshape(batch, height, width)) ** 2 + (
                    y_reprojected - y_ref.reshape(batch, height, width)) ** 2)

        # check |d_reproj-d_1| / d_1 < 0.01
        depth_diff = torch.abs(depth_reprojected - depth_ref)
        relative_depth_diff = depth_diff / depth_ref
        geo_mask = (dist < pixel_thres) & (relative_depth_diff[:, 0] < depth_thres)

    return geo_mask  # B H W


def entropy(volume, dim, keepdim=False):
    return torch.sum(-volume * volume.clamp(1e-9, 1.).log(), dim=dim, keepdim=keepdim)




class convex_upsample_layer_v1_com(nn.Module):
    def __init__(self, feature_dim, scale=2):
        super(convex_upsample_layer_v1_com, self).__init__()
        self.scale = scale
        self.upsample_mask = nn.Sequential(
            nn.Conv2d(feature_dim, 64, 3, stride=1, padding=1, dilation=1, bias=False),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, (2**scale)**2*9, 1, stride=1, padding=0, dilation=1, bias=False)
        )
        # self.q_proj = nn.Linear(144, 144)
        # self.k_proj = nn.Linear(144, 144)

        self.k_proj_1x1 = nn.Conv2d(144, 144, kernel_size=1, bias=True)
        self.q_proj_1x1 = nn.Conv2d(144, 144, kernel_size=1, bias=True)

        #self.norm = nn.GroupNorm(num_groups=9, num_channels=144, affine=False)
        #self.norm =  nn.BatchNorm2d(144)
        self.rope = RoPE(144)
        self.rope._device_weight_init()

    def forward(self, depth, feat=None,k=9):
        feat = torch.randn(1, 64, 48, 160).to(depth.device)
        mask = self.upsample_mask(feat)

        # Apply Positional Encoding
        coords = create_coordinate(mask.shape[-2], mask.shape[-1])
        b, c, h, w = mask.shape
        mask = rearrange(mask, "b c h w -> b (h w) c")
        mask = self.rope(mask, coords)
        mask = rearrange(mask, "b (h w) c -> b c h w", h=h)
       # mask = feature_add_position(mask, 2, 144)

        # Process Queries
        # queries = self.q_proj(mask.view(b, c, -1).permute(0, 2, 1)
        #                                ).reshape(b *h*w,16, 1, 9) #[B*H*W, 16, 1,9]

       # queries = self.norm(queries).view(b, c, -1).permute(0, 2, 1).reshape(b * h * w, 16, 1,9) # [B*H*W, 16, 1,9]
        #queries = queries.view(b, c, -1).permute(0, 2, 1).reshape(b * h * w, 16, 1, 9)  # [B*H*W, 16, 1,9]

        x = self.q_proj_1x1(mask)  # (b, 144, h, w)
        #x = self.norm(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # (b, h, w, 144)
        queries = x.view(b * h * w, 16, 1, 9)  # (b*h*w, 16, 1, 9)  分成16组，每组9个通道

        # Process Keys and Values.
        keys = self.k_proj_1x1(mask)
        #keys = self.norm(keys)
        #mask
        keys = keys.view(-1, 9, h, w) #(b*16,9,h,w)
        keys_window = F.unfold(keys, kernel_size=k,
                                   padding=(k-1)//2)  # [b*16, 9*(3)^2),h*w] 滑动了H*W次，也就是19200次，每次滑动的窗口大小为c*K^2

        # keys_window = keys_window.view(b, 16,9, 3 ** 2, h*w).permute(
        #     0, 4, 1, 3, 2).view(b * h * w, 16,9, 3 ** 2)  # (B, HW, heads, C, KK)

        kw = keys_window.view(b, 16, 9 * k**2, h * w)  # (B,16,81,HW)
        kw = kw.view(b, 16, 9, k**2, h * w)  # (B,16,9,9,HW)
        kw = kw.permute(0, 4, 1, 2, 3).contiguous()  # (B,HW,16,9,9)
        keys_window = kw.view(b * h* w, 16, 9, k**2)  # (BHW,16,9,9)

        scores = torch.matmul(queries, keys_window) / (9 ** 0.5)

        prob = torch.softmax(scores.squeeze(2), dim=-1)

        # keys = self.sft_key(keys, self.key_features_encoder(F.normalize(features, dim=1)))

        return convex_upsample_v1_com(depth, prob, self.scale,k=k)  # B H2 W2

def convex_upsample_v1_com(depth, mask, scale=2,k=None):
    if len(depth.shape) == 3:
        B, H, W = depth.shape
        depth = depth.unsqueeze(1)
    else:
        B, _, H, W = depth.shape

    mask = mask.contiguous().view(B, H, W, 4, 4, k**2)  # (B,H,W,4,4,9)
    mask = mask.permute(0, 5, 3, 4, 1, 2).contiguous()  # (B, 9, 4, 4, H, W)


    up_ = F.unfold(depth, [k,k], padding=(k-1)//2)
    up_ = up_.view(B, k**2, 1, 1, H, W)

    up_ = torch.sum(mask * up_, dim=1)  # B, 2**scale, 2**scale, H, W
    up_ = up_.permute(0, 3, 1, 4, 2).contiguous()    # B H 2**scale W 2**scale
    return up_.reshape(B, 2**scale*H, 2**scale*W)



def profile_once_up(encoder, x):
    flops_e, params_e = profile(encoder, inputs=(x,), verbose=False)

    flops, params = clever_format([flops_e, params_e], "%.3f")

    return flops, params



def get_gpu_mem_info(gpu_id=0):
    """
    根据显卡 id 获取显存使用信息, 单位 MB
    :param gpu_id: 显卡 ID
    :return: total 所有的显存，used 当前使用的显存, free 可使用的显存
    """

    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(r'gpu_id {} 对应的显卡不存在!'.format(gpu_id))
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    return total, used, free







