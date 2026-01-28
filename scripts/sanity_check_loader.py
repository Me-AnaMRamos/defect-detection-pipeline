import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.dataset import MVTecBinaryDataset

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

INDEX_CSV = "data/processed/dataset_index.csv"

train_ds = MVTecBinaryDataset(
    index_csv=INDEX_CSV,
    split="train",
    transform=transform,
)

test_ds = MVTecBinaryDataset(
    index_csv=INDEX_CSV,
    split="test",
    transform=transform,
)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=False)
images, labels = next(iter(train_loader))

# Shape check (B, C, H, W)
assert images.ndim == 4
assert images.shape[1] == 3  # RGB
assert images.shape[2] == 224
assert images.shape[3] == 224

# Label check
assert set(labels.tolist()).issubset({0, 1})

grid = vutils.make_grid(images, nrow=4, normalize=True)

plt.figure(figsize=(8, 4))
plt.imshow(grid.permute(1, 2, 0))
plt.axis("off")

output_path = "outputs/sanity/train_batch_grid.png"
plt.savefig(output_path, bbox_inches="tight")
plt.close()
