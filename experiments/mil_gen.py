#!/usr/bin/env python3
"""mil_gen.py — canonical generator for the ANE op-acceptance survey.

This is the only entry point.  It writes every case (by default in opcases/)
and then removes generated files that are no longer produced (stale
.mil/.meta/.wblob).  Weighted / vision cases are delegated to
weighted_cases.py, which imports the helpers below.

MIL conventions are the textual subset proven to compile on the private
_ANEInMemoryModel path: program(1.3) / func main<ios18> / const() params /
[name = string("..")].

Run:  python3 experiments/mil_gen.py
      python3 experiments/mil_gen.py --output /tmp/opcases

The generator uses only the Python standard library: weights are packed with
struct (IEEE binary16 via "<e", signed int8, and low-nibble-first uint4), so
generation is deterministic and free of numpy.

Weight blobs follow the standard CoreML v2 container (see build_blob):
  file header   +0 uint32 blob_count = actual number of chunks, +4 uint32 version = 2
  chunk header  +0 uint32 0xDEADBEEF
                +4 uint32 dtype tag (1 = fp16, 4 = int8, 0x0b = uint4)
                +8 uint64 data_size in bytes
               +16 uint64 absolute payload offset from start of file
  Chunks start on a 64-byte boundary; BLOBFILE offset points at the chunk
  header, not at the payload.
"""
import argparse
import os
import struct
import sys

sys.dont_write_bytecode = True  # keep opcases regeneration free of __pycache__

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "opcases")
GENERATED = []
BLOB_DTYPE_TAG = {"fp16": 1, "int8": 4, "uint4": 0x0B}

HDR = ('program(1.3)\n'
       '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, '
       '{"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, '
       '{"coremltools-version", "9.0"}})]\n{\n')


# --------------------------------------------------------------------------
# MIL const helpers
# --------------------------------------------------------------------------
def C_str(n, v):
    return f'        string {n} = const()[name = string("{n}"), val = string("{v}")];'


def C_bool(n, v):
    return f'        bool {n} = const()[name = string("{n}"), val = bool({str(v).lower()})];'


def C_fp16(n, v):
    return f'        fp16 {n} = const()[name = string("{n}"), val = fp16({v})];'


def C_i32(n, v):
    return f'        int32 {n} = const()[name = string("{n}"), val = int32({v})];'


def C_fp32(n, v):
    return f'        fp32 {n} = const()[name = string("{n}"), val = fp32({v})];'


def C_i32t(n, t, v):
    vals = ", ".join(str(x) for x in v)
    return (f'        tensor<int32, [{t}]> {n} = const()[name = string("{n}"), '
            f'val = tensor<int32, [{t}]>([{vals}])];')


def C_fp16t(n, shape, v):
    sh = ", ".join(str(x) for x in shape)
    vals = ", ".join(str(x) for x in v)
    return (f'        tensor<fp16, [{sh}]> {n} = const()[name = string("{n}"), '
            f'val = tensor<fp16, [{sh}]>([{vals}])];')


# --------------------------------------------------------------------------
# Weight blob container (v2)
# --------------------------------------------------------------------------
def _flat(values):
    """Yield scalar values from a possibly nested sequence."""
    for value in values:
        if isinstance(value, (list, tuple)):
            yield from _flat(value)
        else:
            yield value


def _pack_fp16(values):
    out = bytearray()
    for value in _flat(values):
        number = float(value)
        try:
            out += struct.pack("<e", number)
        except OverflowError:
            # Match numpy's saturating conversion for values beyond binary16.
            out += struct.pack("<H", 0xFC00 if number < 0 else 0x7C00)
    return bytes(out)


def _pack_int8(values):
    # int() truncates toward zero and the mask reproduces numpy's int8 wrap.
    return bytes((int(value) & 0xFF) for value in _flat(values))


def _pack_uint4(values):
    # Packed two values per byte, low nibble first.
    flat = [int(value) & 0xFF for value in _flat(values)]
    if len(flat) % 2:
        flat.append(0)
    return bytes((flat[i] & 0x0F) | ((flat[i + 1] & 0x0F) << 4)
                 for i in range(0, len(flat), 2))


_PACKERS = {"fp16": _pack_fp16, "int8": _pack_int8, "uint4": _pack_uint4}


def build_blob(tensors, dtype="fp16"):
    """Build a standard v2 blob.  tensors: list of flat value sequences.

    Returns (bytes, [chunk_offset_i]) where chunk_offset_i is the BLOBFILE
    offset (start of the 64-byte chunk header) for each tensor.
    """
    if dtype not in BLOB_DTYPE_TAG:
        raise ValueError(f"unsupported blob dtype {dtype}")
    tag = BLOB_DTYPE_TAG[dtype]
    pack = _PACKERS[dtype]
    buf = bytearray(64)
    struct.pack_into("<I", buf, 0, len(tensors))  # blob count = actual chunks
    struct.pack_into("<I", buf, 4, 2)   # container version
    offs = []
    for t in tensors:
        data = pack(t)
        if len(buf) % 64:
            buf.extend(b"\x00" * (64 - len(buf) % 64))
        off = len(buf)
        desc = bytearray(64)
        desc[0:4] = b"\xef\xbe\xad\xde"
        struct.pack_into("<I", desc, 4, tag)
        struct.pack_into("<Q", desc, 8, len(data))      # data_size
        struct.pack_into("<Q", desc, 16, off + 64)      # absolute data offset
        buf.extend(desc)
        buf.extend(data)
        offs.append(off)
    return bytes(buf), offs


def blof(name, off, shape, dtype="fp16"):
    sh = ", ".join(map(str, shape))
    return (f'        tensor<{dtype}, [{sh}]> {name} = const()[name = string("{name}"), '
            f'val = tensor<{dtype}, [{sh}]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), '
            f'offset = uint64({off})))];')


def _nbytes(shape, elemsize=2):
    n = 1
    for d in shape:
        n *= d
    return n * elemsize


def emit(name, in_shape, out_shape, consts, ops, blobs=None, blob_dtype="fp16"):
    """Write opcases/<name>.mil, .meta and optional .wblob."""
    os.makedirs(OUT, exist_ok=True)
    insh = ", ".join(str(x) for x in in_shape)
    lines = [HDR]
    lines.append(f'    func main<ios18>(tensor<fp16, [{insh}]> x) {{\n')
    lines.extend(c + "\n" for c in consts)
    lines.extend(o + "\n" for o in ops)
    lines.append('    } -> (out);\n}\n')
    with open(os.path.join(OUT, name + ".mil"), "w") as f:
        f.write("".join(lines))

    wf = "-"
    path = os.path.join(OUT, name + ".wblob")
    if blobs:
        data, _ = build_blob(blobs, dtype=blob_dtype)
        with open(path, "wb") as f:
            f.write(data)
        # Record the weight path relative to the output directory's parent.
        # For the default opcases/ this is exactly "opcases/<name>.wblob";
        # an isolated --output DIR yields "<DIR-basename>/<name>.wblob".
        wf = os.path.relpath(path, os.path.dirname(os.path.normpath(OUT)) or ".")
    elif os.path.exists(path):
        os.remove(path)  # this case no longer carries weights
    with open(os.path.join(OUT, name + ".meta"), "w") as f:
        f.write(f"{name},{_nbytes(in_shape)},{_nbytes(out_shape)},{wf}\n")
    GENERATED.append(name)


# --------------------------------------------------------------------------
# Base cases: elementwise / shape / pooling / reduction
# --------------------------------------------------------------------------
IS = [1, 3, 32, 32]
SAME = IS


def build_base_cases():
    # ---- elementwise / activations (output same shape) -------------------
    unary = {
        "relu":      "relu(x = x)",
        "sigmoid":   "sigmoid(x = x)",
        "tanh":      "tanh(x = x)",
        "silu":      "silu(x = x)",
        "relu6":     "relu6(x = x)",
        "softplus":  "softplus(x = x)",
        "sign":      "sign(x = x)",
        "abs":       "abs(x = x)",
        "floor":     "floor(x = x)",
        "ceil":      "ceil(x = x)",
        "round":     "round(x = x)",
        "square":    "square(x = x)",
        "sqrt":      "sqrt(x = x)",
        "exp":       "exp(x = x)",
        "erf":       "erf(x = x)",
        "l2_norm":   "l2_norm(x = x, epsilon = eps)",
    }
    for n, expr in unary.items():
        consts = [C_fp16("eps", "0x1p-5")] if n == "l2_norm" else []
        emit(n, IS, SAME, consts,
             [f'        tensor<fp16, [1, 3, 32, 32]> out = {expr}[name = string("out")];'])

    emit("rsqrt", IS, SAME, [],
         ['        tensor<fp16, [1, 3, 32, 32]> out = rsqrt(x = x)[name = string("out")];'])
    emit("gelu", IS, SAME, [C_str("gm", "EXACT")],
         ['        tensor<fp16, [1, 3, 32, 32]> out = gelu(x = x, mode = gm)[name = string("out")];'])
    emit("leaky_relu", IS, SAME, [C_fp16("al", "0x1p-3")],
         ['        tensor<fp16, [1, 3, 32, 32]> out = leaky_relu(x = x, alpha = al)[name = string("out")];'])
    emit("elu", IS, SAME, [C_fp16("al", "0x1p-3")],
         ['        tensor<fp16, [1, 3, 32, 32]> out = elu(x = x, alpha = al)[name = string("out")];'])
    emit("clip", IS, SAME, [C_fp16("lo", "0x1p-3"), C_fp16("hi", "0x1p2")],
         ['        tensor<fp16, [1, 3, 32, 32]> out = clip(x = x, alpha = lo, beta = hi)[name = string("out")];'])
    emit("softmax", IS, SAME, [C_i32("ax", -1)],
         ['        tensor<fp16, [1, 3, 32, 32]> out = softmax(x = x, axis = ax)[name = string("out")];'])
    # prelu alpha is rank-1 [C], not [C,1,1].
    emit("prelu", IS, SAME, [C_fp16t("al", [3], ["0.1", "0.1", "0.1"])],
         ['        tensor<fp16, [1, 3, 32, 32]> out = prelu(x = x, alpha = al)[name = string("out")];'])

    # ---- binary ----------------------------------------------------------
    for n, op in [("add", "add"), ("sub", "sub"), ("mul", "mul"),
                  ("real_div", "real_div"), ("pow", "pow"), ("maximum", "maximum"),
                  ("minimum", "minimum"), ("mod", "mod")]:
        emit(n, IS, SAME, [],
             [f'        tensor<fp16, [1, 3, 32, 32]> y = mul(x = x, y = x)[name = string("y")];',
              f'        tensor<fp16, [1, 3, 32, 32]> out = {op}(x = x, y = y)[name = string("out")];'])

    # logical ops take bool tensors; cast the result back to fp16 for output.
    emit("logical_xor", IS, SAME,
         [C_fp16("z", "0x0p0"), C_str("dt", "fp16")],
         ['        tensor<bool, [1, 3, 32, 32]> g1 = greater(x = x, y = z)[name = string("g1")];',
          '        tensor<bool, [1, 3, 32, 32]> g2 = less(x = x, y = z)[name = string("g2")];',
          '        tensor<bool, [1, 3, 32, 32]> lb = logical_xor(x = g1, y = g2)[name = string("lb")];',
          '        tensor<fp16, [1, 3, 32, 32]> out = cast(dtype = dt, x = lb)[name = string("out")];'])

    # ---- shape ops -------------------------------------------------------
    emit("transpose_nhwc", IS, [1, 32, 32, 3], [C_i32t("pm", 4, [0, 2, 3, 1])],
         ['        tensor<fp16, [1, 32, 32, 3]> out = transpose(x = x, perm = pm)[name = string("out")];'])
    emit("transpose_hw", IS, SAME, [C_i32t("pm", 4, [0, 1, 3, 2])],
         ['        tensor<fp16, [1, 3, 32, 32]> out = transpose(x = x, perm = pm)[name = string("out")];'])
    emit("reshape", IS, [1, 3, 1, 1024], [],
         ['        tensor<fp16, [1, 3, 1, 1024]> out = reshape(x = x, shape = tensor<int32, [4]>([1, 3, 1, 1024]))[name = string("out")];'])
    emit("concat_ch", IS, [1, 6, 32, 32], [C_i32("ax", 1), C_bool("il", False)],
         ['        tensor<fp16, [1, 6, 32, 32]> out = concat(axis = ax, interleave = il, values = (x, x))[name = string("out")];'])
    emit("slice_by_size", IS, [1, 3, 16, 16],
         [C_i32t("bg", 4, [0, 0, 0, 0]), C_i32t("sz", 4, [1, 3, 16, 16])],
         ['        tensor<fp16, [1, 3, 16, 16]> out = slice_by_size(x = x, begin = bg, size = sz)[name = string("out")];'])
    emit("tile", IS, [1, 3, 64, 64], [C_i32t("rp", 4, [1, 1, 2, 2])],
         ['        tensor<fp16, [1, 3, 64, 64]> out = tile(x = x, reps = rp)[name = string("out")];'])
    emit("crop", IS, [1, 3, 28, 28],
         [C_i32t("ch", 2, [2, 2]), C_i32t("cw", 2, [2, 2])],
         ['        tensor<fp16, [1, 3, 28, 28]> out = crop(x = x, crop_height = ch, crop_width = cw)[name = string("out")];'])

    # pad is per-dimension (before, after) pairs in N,C,H,W order.  For
    # [1,3,32,32] the correct 1-pixel H/W padding is [0,0,0,0,1,1,1,1].
    for mode in ("constant", "reflect", "replicate"):
        emit(f"pad_{mode}", IS, [1, 3, 34, 34],
             [C_i32t("pd", 8, [0, 0, 0, 0, 1, 1, 1, 1]), C_str("md", mode),
              C_fp16("cv", "0x0p0")],
             [f'        tensor<fp16, [1, 3, 34, 34]> out = pad(x = x, pad = pd, mode = md, constant_val = cv)[name = string("out")];'])
    # two rows of causal padding on H.
    emit("pad_causal", IS, [1, 3, 34, 32],
         [C_i32t("pd", 8, [0, 0, 0, 0, 2, 0, 0, 0]), C_str("md", "constant"),
          C_fp16("cv", "0x0p0")],
         ['        tensor<fp16, [1, 3, 34, 32]> out = pad(x = x, pad = pd, mode = md, constant_val = cv)[name = string("out")];'])

    # ---- pooling / reduce -------------------------------------------------
    emit("max_pool_3x3", IS, [1, 3, 15, 15],
         [C_i32t("ks", 2, [3, 3]), C_i32t("st", 2, [2, 2]), C_str("pt", "valid"),
          C_i32t("pd", 4, [0, 0, 0, 0]), C_bool("cm", False)],
         ['        tensor<fp16, [1, 3, 15, 15]> out = max_pool(x = x, kernel_sizes = ks, strides = st, pad_type = pt, pad = pd, ceil_mode = cm)[name = string("out")];'])
    emit("max_pool_2x2", IS, [1, 3, 16, 16],
         [C_i32t("ks", 2, [2, 2]), C_i32t("st", 2, [2, 2]), C_str("pt", "valid"),
          C_i32t("pd", 4, [0, 0, 0, 0]), C_bool("cm", False)],
         ['        tensor<fp16, [1, 3, 16, 16]> out = max_pool(x = x, kernel_sizes = ks, strides = st, pad_type = pt, pad = pd, ceil_mode = cm)[name = string("out")];'])
    emit("avg_pool_3x3", IS, [1, 3, 15, 15],
         [C_i32t("ks", 2, [3, 3]), C_i32t("st", 2, [2, 2]), C_str("pt", "valid"),
          C_i32t("pd", 4, [0, 0, 0, 0]), C_bool("cm", False), C_bool("ep", False)],
         ['        tensor<fp16, [1, 3, 15, 15]> out = avg_pool(x = x, kernel_sizes = ks, strides = st, pad_type = pt, pad = pd, ceil_mode = cm, exclude_padding_from_average = ep)[name = string("out")];'])
    emit("avg_pool_same", IS, [1, 3, 16, 16],
         [C_i32t("ks", 2, [3, 3]), C_i32t("st", 2, [2, 2]), C_str("pt", "same"),
          C_i32t("pd", 4, [0, 0, 0, 0]), C_bool("cm", False), C_bool("ep", False)],
         ['        tensor<fp16, [1, 3, 16, 16]> out = avg_pool(x = x, kernel_sizes = ks, strides = st, pad_type = pt, pad = pd, ceil_mode = cm, exclude_padding_from_average = ep)[name = string("out")];'])

    for nm in ("reduce_mean", "reduce_max", "reduce_sum", "reduce_min", "reduce_prod"):
        emit(nm, IS, [1, 3, 1, 1], [C_i32t("ax", 2, [2, 3]), C_bool("kd", True)],
             [f'        tensor<fp16, [1, 3, 1, 1]> out = {nm}(x = x, axes = ax, keep_dims = kd)[name = string("out")];'])
    emit("reduce_mean_noch", [1, 32, 32, 3], [1, 1, 1, 3],
         [C_i32t("ax", 3, [0, 1, 2]), C_bool("kd", True)],
         ['        tensor<fp16, [1, 1, 1, 3]> out = reduce_mean(x = x, axes = ax, keep_dims = kd)[name = string("out")];'])

    # ---- rank edge case ---------------------------------------------------
    emit("expand_dims", IS, [1, 1, 3, 32, 32], [C_i32t("ax", 1, [0])],
         ['        tensor<fp16, [1, 1, 3, 32, 32]> out = expand_dims(x = x, axes = ax)[name = string("out")];'])


def clean_stale():
    """Remove generated files that the current generators no longer produce."""
    keep = set(GENERATED)
    removed = []
    for fn in sorted(os.listdir(OUT)):
        stem, ext = os.path.splitext(fn)
        if ext in (".mil", ".meta", ".wblob") and stem not in keep:
            os.remove(os.path.join(OUT, fn))
            removed.append(fn)
    with open(os.path.join(OUT, "_index.txt"), "w") as f:
        for name in sorted(keep):
            f.write(name + "\n")
    return removed


def main(argv=None):
    global OUT
    parser = argparse.ArgumentParser(
        description="Generate the canonical ANE op-acceptance cases.")
    parser.add_argument("--output", metavar="DIR", default=None,
                        help="output directory (default: experiments/opcases)")
    args = parser.parse_args(argv)
    if args.output:
        OUT = os.path.abspath(args.output)
    build_base_cases()
    # When run as a script this module is "__main__"; make the canonical
    # helpers importable by weighted_cases without loading a second copy of
    # this module (which would have its own GENERATED/OUT state).
    sys.modules["mil_gen"] = sys.modules[__name__]
    import weighted_cases  # delegated weighted / vision cases
    weighted_cases.build_cases()
    removed = clean_stale()
    cases = sorted(GENERATED)
    print(f"generated {len(cases)} cases in {os.path.relpath(OUT, os.getcwd())}")
    if removed:
        print(f"removed {len(removed)} stale generated files: "
              + ", ".join(sorted({os.path.splitext(f)[0] for f in removed})))


if __name__ == "__main__":
    main()
