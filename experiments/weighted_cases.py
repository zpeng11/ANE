#!/usr/bin/env python3
"""weighted_cases.py — weighted / vision cases for the ANE acceptance survey.

This module is delegated to by mil_gen.py (the canonical entry point).  All
shared emit/blob helpers come from mil_gen, so there is exactly one writer for
opcases/ and one cleanup pass for stale files.

Do not run this file directly; use:  python3 experiments/mil_gen.py
"""
import random

from mil_gen import (
    C_fp16,
    C_fp32,
    C_i32,
    C_i32t,
    C_str,
    blof,
    build_blob,
    emit,
)


def wts(n, scale=0.05):
    r = random.Random(7)
    return [r.uniform(-scale, scale) for _ in range(n)]


def conv_case(name, in_ch, out_ch, kh, kw, groups, pad_type, dil, out_hw):
    wshape = [out_ch, in_ch // groups, kh, kw]
    n = 1
    for d in wshape:
        n *= d
    blobs = [wts(n)]
    _, offs = build_blob(blobs)
    cons = [
        C_i32t("dl", 2, list(dil)),
        C_i32("gp", groups),
        C_i32t("pd", 4, [0, 0, 0, 0]),
        C_str("pt", pad_type),
        C_i32t("st", 2, [1, 1]),
        blof("W", offs[0], wshape),
    ]
    ops = [f'        tensor<fp16, [1, {out_ch}, {out_hw[0]}, {out_hw[1]}]> out = '
           f'conv(dilations = dl, groups = gp, pad = pd, pad_type = pt, '
           f'strides = st, weight = W, x = x)[name = string("out")];']
    emit(name, [1, in_ch, 32, 32], [1, out_ch, out_hw[0], out_hw[1]], cons, ops, blobs=blobs)


def build_cases():
    # ---- conv variants ----------------------------------------------------
    conv_case("conv_1x1", 3, 4, 1, 1, 1, "valid", (1, 1), (32, 32))
    conv_case("conv_3x3_same", 3, 4, 3, 3, 1, "same", (1, 1), (32, 32))
    conv_case("conv_3x3_valid", 3, 4, 3, 3, 1, "valid", (1, 1), (30, 30))
    conv_case("conv_dw_3x3", 3, 3, 3, 3, 3, "same", (1, 1), (32, 32))
    conv_case("conv_5x5_same", 3, 8, 5, 5, 1, "same", (1, 1), (32, 32))
    conv_case("conv_dilated", 3, 4, 3, 3, 1, "valid", (2, 2), (28, 28))
    conv_case("conv_1x1_pad", 3, 4, 1, 1, 1, "same", (1, 1), (32, 32))

    # ---- conv_transpose ---------------------------------------------------
    wt = wts(3 * 1 * 3 * 3)
    _, offs = build_blob([wt])
    emit("conv_transpose", [1, 3, 32, 32], [1, 3, 64, 64],
         [C_i32t("dl", 2, [1, 1]), C_i32("gp", 3), C_i32t("pd", 4, [0, 0, 0, 0]),
          C_str("pt", "same"), C_i32t("st", 2, [2, 2]),
          C_i32t("os", 4, [1, 3, 64, 64]), blof("W", offs[0], [3, 1, 3, 3])],
         ['        tensor<fp16, [1, 3, 64, 64]> out = conv_transpose(dilations = dl, groups = gp, pad = pd, pad_type = pt, strides = st, output_shape = os, weight = W, x = x)[name = string("out")];'],
         blobs=[wt])

    # ---- normalization ----------------------------------------------------
    for name in ("batch_norm", "instance_norm"):
        blobs = [wts(3, 0.1), [1.0] * 3, [1.0] * 3, [0.0] * 3]
        _, offs = build_blob(blobs)
        cons = [C_fp16("ep", "0x1.0c6f7a0b5ed8dp-20"),
                blof("W", offs[0], [3]), blof("W2", offs[1], [3]),
                blof("W3", offs[2], [3]), blof("W4", offs[3], [3])]
        if name == "batch_norm":
            ops = ['        tensor<fp16, [1, 3, 32, 32]> out = batch_norm(x = x, mean = W, variance = W2, gamma = W3, beta = W4, epsilon = ep)[name = string("out")];']
        else:
            ins = ('        tensor<fp16, [1, 3, 32, 32]> out = '
                   'instance_norm(x = x, gamma = W3, beta = W4, epsilon = ep)[name = string("out")];')
            ops = [ins]
        emit(name, [1, 3, 32, 32], [1, 3, 32, 32], cons, ops, blobs=blobs)

    # ---- resize / upsample ------------------------------------------------
    emit("resize_nn", [1, 3, 8, 8], [1, 3, 16, 16],
         [C_i32("th", 16), C_i32("tw", 16)],
         ['        tensor<fp16, [1, 3, 16, 16]> out = resize_nearest_neighbor(x = x, target_size_height = th, target_size_width = tw)[name = string("out")];'])
    emit("resize_bilinear", [1, 3, 8, 8], [1, 3, 16, 16],
         [C_i32("th", 16), C_i32("tw", 16), C_str("sm", "DEFAULT")],
         ['        tensor<fp16, [1, 3, 16, 16]> out = resize_bilinear(x = x, target_size_height = th, target_size_width = tw, sampling_mode = sm)[name = string("out")];'])
    emit("upsample_nn", [1, 3, 8, 8], [1, 3, 16, 16],
         [C_fp32("sh", 2.0), C_fp32("sw", 2.0)],
         ['        tensor<fp16, [1, 3, 16, 16]> out = upsample_nearest_neighbor(x = x, scale_factor_height = sh, scale_factor_width = sw)[name = string("out")];'])
    emit("upsample_bilinear", [1, 3, 8, 8], [1, 3, 16, 16],
         [C_fp32("sh", 2.0), C_fp32("sw", 2.0)],
         ['        tensor<fp16, [1, 3, 16, 16]> out = upsample_bilinear(x = x, scale_factor_height = sh, scale_factor_width = sw)[name = string("out")];'])

    # ---- select / logical -------------------------------------------------
    # select takes (cond, a, b); logical ops take bool tensors.
    emit("select", [1, 3, 32, 32], [1, 3, 32, 32],
         [C_fp16("z", "0x0p0")],
         ['        tensor<bool, [1, 3, 32, 32]> g = greater(x = x, y = z)[name = string("g")];',
          '        tensor<fp16, [1, 3, 32, 32]> out = select(cond = g, a = x, b = z)[name = string("out")];'])
    emit("logical_and_select", [1, 3, 32, 32], [1, 3, 32, 32],
         [C_fp16("z", "0x0p0"), C_fp16("one", "0x1p0")],
         ['        tensor<bool, [1, 3, 32, 32]> g1 = greater(x = x, y = z)[name = string("g1")];',
          '        tensor<bool, [1, 3, 32, 32]> g2 = less(x = x, y = one)[name = string("g2")];',
          '        tensor<bool, [1, 3, 32, 32]> lb = logical_and(x = g1, y = g2)[name = string("lb")];',
          '        tensor<fp16, [1, 3, 32, 32]> out = select(cond = lb, a = x, b = z)[name = string("out")];'])

    # ---- gather -----------------------------------------------------------
    # gather output shape replaces `axis` with the indices shape.
    # validate_indices is left at its default (False) to keep the grammar minimal.
    emit("gather_axis0", [1, 3, 32, 32], [2, 3, 32, 32],
         [C_i32t("ix", 2, [0, 0]), C_i32("ax", 0)],
         ['        tensor<fp16, [2, 3, 32, 32]> out = gather(x = x, indices = ix, axis = ax)[name = string("out")];'])
    emit("gather_axis1", [1, 4, 8, 8], [1, 2, 8, 8],
         [C_i32t("ix", 2, [0, 2]), C_i32("ax", 1)],
         ['        tensor<fp16, [1, 2, 8, 8]> out = gather(x = x, indices = ix, axis = ax)[name = string("out")];'])
    # gather_along_axis requires rank(indices) == rank(x); output == indices shape.
    idx = [i % 3 for i in range(1 * 2 * 8 * 8)]
    emit("gather_along_axis", [1, 3, 8, 8], [1, 2, 8, 8],
         [C_i32t("ix", "1, 2, 8, 8", idx), C_i32("ax", 1)],
         ['        tensor<fp16, [1, 2, 8, 8]> out = gather_along_axis(x = x, indices = ix, axis = ax)[name = string("out")];'])

    # ---- pixel shuffle / depth_to_space ----------------------------------
    emit("depth_to_space4", [1, 12, 8, 8], [1, 3, 16, 16],
         [C_i32("bs", 2)],
         ['        tensor<fp16, [1, 3, 16, 16]> out = depth_to_space(x = x, block_size = bs)[name = string("out")];'])
    emit("pixel_shuffle4", [1, 12, 8, 8], [1, 3, 16, 16],
         [C_i32("uf", 2)],
         ['        tensor<fp16, [1, 3, 16, 16]> out = pixel_shuffle(x = x, upscale_factor = uf)[name = string("out")];'])

    # ---- slice_by_index (bool masks) -------------------------------------
    emit("slice_by_index", [1, 3, 32, 32], [1, 3, 16, 16],
         [C_i32t("bg", 4, [0, 0, 0, 0]), C_i32t("en", 4, [1, 3, 16, 16]),
          C_i32t("sd", 4, [1, 1, 1, 1]),
          '        tensor<bool, [4]> bm = const()[name = string("bm"), val = tensor<bool, [4]>([false, false, false, false])];',
          '        tensor<bool, [4]> em = const()[name = string("em"), val = tensor<bool, [4]>([false, false, false, false])];',
          '        tensor<bool, [4]> sq = const()[name = string("sq"), val = tensor<bool, [4]>([false, false, false, false])];'],
         ['        tensor<fp16, [1, 3, 16, 16]> out = slice_by_index(x = x, begin = bg, end = en, stride = sd, begin_mask = bm, end_mask = em, squeeze_mask = sq)[name = string("out")];'])


if __name__ == "__main__":
    raise SystemExit("weighted_cases.py is a case module; run: python3 experiments/mil_gen.py")
