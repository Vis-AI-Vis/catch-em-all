from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class PokemonDataset(Dataset):
    EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

    def __init__(self, root, processor, label2id=None, name=""):
        """
        If label2id is None, all images are loaded with label id -1
        (e.g. for animals, which are not classes of the model).
        """
        self.processor = processor
        self.samples = []
        for path in sorted(Path(root).rglob("*")):
            if path.suffix.lower() not in self.EXTENSIONS:
                continue
            if label2id is None:
                self.samples.append((path, -1))
            elif path.parent.name in label2id:
                self.samples.append((path, label2id[path.parent.name]))
        self.name = name

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_id = self.samples[idx]
        img = Image.open(path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0), label_id, idx

    def get_paths(self, indices):
        return [self.samples[i][0] for i in indices]
