"""Batch translation script using trained Attention-CycleGAN.

Translates labeled C-NMC source images into ALL-IDB target domain style
preserving labels for subsequent ResNet34 classifier training.
"""

from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from src.data.backgrounds import background_for, composite_on_background, load_background_paths
from src.models.attention import AttentionFusionModule
from src.models.generator import AttentionGenerator


def translate_source_dataset(
    checkpoint_path: str,
    source_dir: str = "data/processed/source_cnmc/train",
    output_dir: str = "data/processed/translated_source/train",
    image_size: int = 128,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    background_dir: str | None = None,
) -> tuple[int, int]:
    """Translate source domain dataset into target domain style.

    Args:
        checkpoint_path: Path to trained CycleGAN .pth checkpoint.
        source_dir: Directory containing original source images ('all' and 'hem').
        output_dir: Directory to save translated output images.
        image_size: Resolution of images (default 128).
        device: Target execution device.
        background_dir: Bank of target backgrounds; each source cell is pasted onto a background chosen from its file
            name before translation, matching how the GAN was trained. None translates the raw source image.

    Returns:
        Tuple of (num_all_translated, num_hem_translated).
    """
    target_device = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location=target_device, weights_only=True)
    if ckpt.get("collapsed"):
        print(f"WARNING: {checkpoint_path} collapsed (s' = s): translated images equal the source.")
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
    backgrounds = load_background_paths(background_dir) if background_dir else []

    counts = []
    for cls in ("all", "hem"):
        out_dir = Path(output_dir) / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted((Path(source_dir) / cls).glob("*.png"))
        with torch.no_grad():
            for f in tqdm(files, desc=f"Translating {cls.upper()}"):
                with Image.open(f) as img:
                    source = img.convert("RGB")
                if backgrounds:
                    source = composite_on_background(source, Image.open(background_for(f.name, backgrounds)))
                tensor_img = transform(source).unsqueeze(0).to(target_device)
                s_prime, _ = attn(tensor_img, gen(tensor_img))
                out_tensor = (s_prime.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
                T.ToPILImage()(out_tensor).save(out_dir / f.name)
        counts.append(len(files))

    print(f"Translation finished: {counts[0]} ALL, {counts[1]} HEM saved to {output_dir}")
    return counts[0], counts[1]
