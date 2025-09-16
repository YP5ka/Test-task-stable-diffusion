from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


def build_transform(resolution: int) -> T.Compose:
    return T.Compose(
            [T.Resize(resolution, interpolation=Image.BICUBIC),
             T.CenterCrop(resolution),
             T.ToTensor(),
             T.Normalize([0.5], [0.5]),
             ]
    )


class ImageDataset(Dataset):
    IMAGE_EXTS = {"jpg", "jpeg", "png", "webp"}

    def __init__(
            self,
            root: str | Path,
            resolution: int = 512,
            default_caption_template: str = "default tocken",
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(self.root)

        self.placeholder_token = ""
        self.default_caption_template = default_caption_template
        self.transform = build_transform(resolution)

        self._items: List[Tuple[Path, str]] = self._gather_items()

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        img_path, caption = self._items[idx]
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)
        return {
            "pixel_values": pixel_values,
            "caption": caption,
        }

    def _gather_items(self) -> List[Tuple[Path, str]]:
        captions_global = self._read_global_captions()
        items: List[Tuple[Path, str]] = []
        for img_path in sorted(self.root.iterdir()):
            if img_path.suffix.lstrip(".").lower() not in self.IMAGE_EXTS:
                continue
            caption = self._resolve_caption(img_path, captions_global)
            items.append((img_path, caption))
        if not items:
            raise RuntimeError(f"No training images found in {self.root}")
        return items

    def _read_global_captions(self) -> dict[str, str]:
        file = self.root / "captions.txt"
        if not file.exists():
            return {}
        mapping: dict[str, str] = {}
        for line in file.read_text(encoding="utf-8").splitlines():
            if not line.strip() or "\t" not in line:
                continue
            fname, caption = line.split("\t", maxsplit=1)
            mapping[fname.strip()] = caption.strip()
        return mapping

    def _resolve_caption(self, img_path: Path, global_caps: dict[str, str]) -> str:
        sidecar = img_path.with_suffix(".txt")
        if sidecar.exists():
            return sidecar.read_text(encoding="utf-8").strip()
        if img_path.name in global_caps:
            return global_caps[img_path.name]
        return self.default_caption_template.format(token=self.placeholder_token)
