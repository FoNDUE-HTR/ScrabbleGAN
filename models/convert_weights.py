"""
convert_legacy.py
=================
Convertit le format PyTorch legacy (data.pkl + dossier data/) vers un .pt moderne.

Le format legacy stocke :
  - data.pkl  : structure pickle avec persistent_id référençant des clés numériques
  - data/     : dossier contenant les tenseurs bruts, un fichier par clé

Usage :
    python convert_legacy.py --pkl ./data.pkl --data_dir ./data --output ./data_final.pt
"""

import pickle
import argparse
import io
import struct
import numpy as np
from pathlib import Path
from collections import OrderedDict


class LegacyUnpickler(pickle.Unpickler):
    """
    Charge data.pkl en lisant les tenseurs bruts depuis le dossier data/.
    persistent_id = ('storage', storage_type, key, device, numel)
    key = nom du fichier dans data/ (ex: '181829168')
    """

    def __init__(self, f, data_dir: Path):
        super().__init__(f)
        self._data_dir = data_dir
        self._cache    = {}

    def persistent_load(self, pid):
        import torch

        if not (isinstance(pid, tuple) and pid[0] == 'storage'):
            raise pickle.UnpicklingError(f"persistent_id inconnu : {pid!r}")

        _, storage_type, key, device, numel = pid

        if key in self._cache:
            return self._cache[key]

        dtype_map = {
            'FloatStorage':  (torch.float32, np.float32,  4),
            'DoubleStorage': (torch.float64, np.float64,  8),
            'HalfStorage':   (torch.float16, np.float16,  2),
            'LongStorage':   (torch.int64,   np.int64,    8),
            'IntStorage':    (torch.int32,   np.int32,    4),
            'ShortStorage':  (torch.int16,   np.int16,    2),
            'ByteStorage':   (torch.uint8,   np.uint8,    1),
            'CharStorage':   (torch.int8,    np.int8,     1),
            'BoolStorage':   (torch.bool,    np.bool_,    1),
        }

        type_name = (storage_type.__name__
                     if hasattr(storage_type, '__name__')
                     else str(storage_type).split('.')[-1])
        dtype_torch, dtype_np, itemsize = dtype_map.get(
            type_name, (torch.float32, np.float32, 4)
        )

        blob_path = self._data_dir / key
        if blob_path.exists():
            raw  = blob_path.read_bytes()
            expected = numel * itemsize
            if len(raw) >= expected:
                arr = np.frombuffer(raw[:expected], dtype=dtype_np).copy()
            else:
                print(f"  [!] {key} : {len(raw)} bytes < attendu {expected}, zero-padding")
                arr = np.zeros(numel, dtype=dtype_np)
            t = torch.from_numpy(arr)
        else:
            print(f"  [!] Fichier manquant : {blob_path}, tenseur zero")
            t = torch.zeros(numel, dtype=dtype_torch)

        self._cache[key] = t
        return t

    def find_class(self, module, name):
        # Redirections pour anciens modules
        remap = {
            ('torch._utils', '_rebuild_tensor_v2'): self._rebuild_tensor,
        }
        if (module, name) in remap:
            return remap[(module, name)]
        return super().find_class(module, name)

    @staticmethod
    def _rebuild_tensor(storage, offset, size, stride, requires_grad, hooks, metadata=None):
        import torch
        # storage est un tenseur 1D, on le reshape
        if isinstance(storage, torch.Tensor):
            flat = storage
        else:
            flat = torch.zeros(1)
        t = flat.as_strided(size, stride, offset)
        t = t.detach().clone()
        t.requires_grad = requires_grad
        return t


def convert(pkl_path: str, data_dir: str, output_path: str):
    import torch

    pkl_path    = Path(pkl_path)
    data_dir    = Path(data_dir)
    output_path = Path(output_path)

    print(f"[1/3] Chargement de {pkl_path} avec tenseurs depuis {data_dir}/")
    print(f"  -> {len(list(data_dir.iterdir()))} fichiers tenseurs trouves")

    with open(pkl_path, 'rb') as f:
        unpickler = LegacyUnpickler(f, data_dir)
        data = unpickler.load()

    print(f"[2/3] Analyse du contenu...")
    if isinstance(data, dict):
        print(f"  Cles : {list(data.keys())}")
        state = data.get('model', data)
    else:
        state = data

    # Statistiques
    n_ok  = sum(1 for v in state.values()
                if isinstance(v, torch.Tensor) and v.abs().max() > 0)
    n_zero = sum(1 for v in state.values()
                 if isinstance(v, torch.Tensor) and v.abs().max() == 0)
    n_nan  = sum(1 for v in state.values()
                 if isinstance(v, torch.Tensor) and torch.isnan(v).any())
    print(f"  Tenseurs OK={n_ok}  zeros={n_zero}  NaN={n_nan}")

    # Afficher quelques valeurs pour verifier
    for k, v in list(state.items())[:3]:
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape} min={v.min():.4f} max={v.max():.4f}")

    print(f"[3/3] Sauvegarde -> {output_path}")
    wrapped = {
        'model': OrderedDict(state),
        'epoch': 0,
        'best_fid': 0,
    }
    torch.save(wrapped, output_path)
    print(f"  -> {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Verification
    check = torch.load(output_path, map_location='cpu', weights_only=False)
    n = len(check['model'])
    print(f"\n[OK] {n} tenseurs charges correctement")
    print(f"\n  Relancez :")
    print(f"  python htr_synth_v2.py --step generate --weights {output_path} ...")


def verify(pt_path: str):
    """Vérifie un fichier .pt converti."""
    import torch
    print(f"Vérification de {pt_path}...")
    state = torch.load(pt_path, map_location="cpu", weights_only=False)
    if "model" not in state:
        print("[!] Clé 'model' absente")
        return
    sd = state["model"]
    total   = len(sd)
    nonzero = sum(1 for v in sd.values() if isinstance(v, torch.Tensor) and v.abs().max() > 0)
    print(f"  Tenseurs total    : {total}")
    print(f"  Tenseurs non-nuls : {nonzero}")
    print(f"  Epoch             : {state.get('epoch', 'N/A')}")
    print(f"[OK]" if nonzero > 0 else "[!] Tous nuls — conversion incorrecte")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl',      default='./data.pkl')
    ap.add_argument('--data_dir', default='./data')
    ap.add_argument('--output',   default='./data_final.pt')
    ap.add_argument('--verify',   default=None, help='Vérifier un .pt existant')
    args = ap.parse_args()
    if args.verify:
        verify(args.verify)
    else:
        convert(args.pkl, args.data_dir, args.output)


if __name__ == '__main__':
    main()
