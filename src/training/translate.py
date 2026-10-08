"""Batch translation script using trained Attention-CycleGAN.

Translates labeled C-NMC source images into ALL-IDB target domain style
preserving labels for subsequent ResNet34 classifier training.
"""

from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from src.models.attention import AttentionFusionModule
from src.models.generator import AttentionGenerator


def translate_source_dataset(
    checkpoint_path: str,
    source_dir: str = "data/processed/source_cnmc/train",
    output_dir: str = "data/processed/translated_source/train",
    image_size: int = 128,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[int, int]:
    """Translate source domain dataset into target domain style.

    Args:
        checkpoint_path: Path to trained CycleGAN .pth checkpoint.
        source_dir: Directory containing original source images ('all' and 'hem').
        output_dir: Directory to save translated output images.
        image_size: Resolution of images (default 128).
        device: Target execution device.

    Returns:
        Tuple of (num_all_translated, num_hem_translated).
    """
    target_device = torch.device(device)
    out_all_dir = Path(output_dir) / "all"
    out_hem_dir = Path(output_dir) / "hem"
    out_all_dir.mkdir(parents=True, exist_ok=True)
    out_hem_dir.mkdir(parents=True, exist_ok=True)

    # Load generator and attention module
    ckpt = torch.load(checkpoint_path, map_location=target_device, weights_only=True)
    gen = AttentionGenerator(3, 3, num_res_blocks=6).to(target_device)
    attn = AttentionFusionModule(kernel_size=7).to(target_device)

    gen.load_state_dict(ckpt["gen_s2t_state"])
    attn.load_state_dict(ckpt["attn_s_state"])
    gen.eval()
    attn.eval()

    transform = T.Compose(
        [
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    src_path = Path(source_dir)
    all_files = sorted((src_path / "all").glob("*.png"))
    hem_files = sorted((src_path / "hem").glob("*.png"))

    print(f"Translating {len(all_files)} ALL images using trained Attention-CycleGAN...")
    with torch.no_grad():
        for f in tqdm(all_files, desc="Translating ALL"):
            with Image.open(f) as img:
                tensor_img = transform(img.convert("RGB")).unsqueeze(0).to(target_device)
                g_s = gen(tensor_img)
                s_prime, _ = attn(tensor_img, g_s)

                # Denormalize [-1, 1] to [0, 255]
                out_tensor = (s_prime.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
                out_pil = T.ToPILImage()(out_tensor)
                out_pil.save(out_all_dir / f.name)

    print(f"Translating {len(hem_files)} HEM images using trained Attention-CycleGAN...")
    with torch.no_grad():
        for f in tqdm(hem_files, desc="Translating HEM"):
            with Image.open(f) as img:
                tensor_img = transform(img.convert("RGB")).unsqueeze(0).to(target_device)
                g_s = gen(tensor_img)
                s_prime, _ = attn(tensor_img, g_s)

                out_tensor = (s_prime.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
                out_pil = T.ToPILImage()(out_tensor)
                out_pil.save(out_hem_dir / f.name)

    print(f"Translation finished: {len(all_files)} ALL, {len(hem_files)} HEM saved to {output_dir}")
    return len(all_files), len(hem_files)
