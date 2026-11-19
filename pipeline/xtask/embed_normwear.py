"""NormWear (Apache-2.0, mosaic-laboratory/normwear) embeddings for every 64 Hz task segment.

Each segment [C, L] is cut into non-overlapping 388-sample chunks (6.06 s, the model's pre-training length).
Each chunk/channel is z-scored, all channels of a chunk go through the encoder together (the model fuses
sensors via its shared [CLS] tokens). We keep, per chunk and channel, the [CLS] embedding and the mean of
patch tokens (768-d each, float16).

Model code is vendored from the Hugging Face repo (reviewed: no network/exec calls) and weights are read
from model.safetensors, so no remote code is executed.

Usage: python embed_normwear.py --seg-dir SEGMENTS --model-dir NORMWEAR_DIR --out OUT [--device cuda] [--batch 128]
Output: OUT/emb_<DS>.npz with seg_id, chunk_t0 (s), cls [N, C, 768], mean [N, C, 768]
"""
import os, sys, json, argparse, time
import numpy as np, pandas as pd, torch

CHUNK = 388
FS = 64


def load_model(model_dir, device):
    torch.uint64 = torch.int64; torch.uint32 = torch.int32; torch.uint16 = torch.int16
    sys.path.insert(0, model_dir)
    from nw.normwear import NormWear
    from safetensors.torch import load_file
    cfg = json.load(open(os.path.join(model_dir, 'config.json')))
    keys = ['img_size', 'patch_size', 'in_chans', 'target_len', 'nvar', 'embed_dim', 'decoder_embed_dim', 'depth', 'num_heads',
            'decoder_depth', 'mlp_ratio', 'fuse_freq', 'is_pretrain', 'mask_t_prob', 'mask_f_prob', 'mask_prob', 'mask_scheme',
            'use_cwt', 'attn_score', 'comb_freq', 'no_fusion', 'mean_fuse', 'include_msitf', 'query_size']
    m = NormWear(**{k: (tuple(cfg[k]) if isinstance(cfg[k], list) else cfg[k]) for k in keys})
    missing, unexpected = m.load_state_dict(load_file(os.path.join(model_dir, 'model.safetensors')), strict=False)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    return m.eval().to(device)


def chunks_of(arr):
    C, L = arr.shape
    n = L // CHUNK
    if n == 0:
        return None, None
    x = arr[:, :n * CHUNK].reshape(C, n, CHUNK).transpose(1, 0, 2).astype(np.float32)  # [n, C, CHUNK]
    mu, sd = x.mean(-1, keepdims=True), x.std(-1, keepdims=True)
    valid = (sd[..., 0] > 1e-6).all(1)
    x = np.ascontiguousarray((x - mu) / np.maximum(sd, 1e-6))
    return x, valid


@torch.no_grad()
def embed_dataset(m, seg_dir, ds, out, device, batch):
    idx = pd.read_parquet(os.path.join(seg_dir, f'{ds}_index.parquet'))
    X, meta = [], []
    for r in idx.itertuples():
        arr = np.load(os.path.join(seg_dir, ds, f'{r.seg_id}.npy'))
        x, valid = chunks_of(arr)
        if x is None:
            continue
        for j in np.where(valid)[0]:
            X.append(x[j]); meta.append((r.seg_id, j * CHUNK / FS))
    X = np.stack(X)
    N, C, _ = X.shape
    cls = np.zeros((N, C, 768), np.float16); mean = np.zeros((N, C, 768), np.float16)
    t = time.time()
    use_amp = device.startswith('cuda')
    for i in range(0, N, batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device)
        with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            enc = m(xb, return_spec=False, return_enc_out=True, return_dec_out=False)['enc_out']  # [b, C, 1+P, 768]
        cls[i:i + batch] = enc[:, :, 0, :].float().cpu().numpy()
        mean[i:i + batch] = enc[:, :, 1:, :].float().mean(2).cpu().numpy()
        if (i // batch) % 50 == 0:
            print(f'{ds} {i}/{N} chunks, {time.time()-t:.0f}s', flush=True)
    np.savez(os.path.join(out, f'emb_{ds}.npz'), seg_id=np.array([s for s, _ in meta]), chunk_t0=np.array([c for _, c in meta]),
             cls=cls, mean=mean)
    print(f'{ds}: {N} chunks x {C} channels in {time.time()-t:.0f}s', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seg-dir', required=True); ap.add_argument('--model-dir', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu'); ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--datasets', default='COGBCI,MPHRC,UNIVERSE')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print('device', a.device, torch.__version__, torch.cuda.get_device_name(0) if a.device.startswith('cuda') else '', flush=True)
    m = load_model(a.model_dir, a.device)
    for ds in a.datasets.split(','):
        embed_dataset(m, a.seg_dir, ds, a.out, a.device, a.batch)
