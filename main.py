import torch
import torch.utils.data
import torchvision
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 32

class DatasetCIFAR10(torch.utils.data.Dataset):

    def __init__(self, is_train):
        super().__init__()

        self.data = torchvision.datasets.cifar.CIFAR10(
            root='./tmp',
            train=is_train,
            download=True
        )

    def __getitem__(self, index):
        x, y_idx = self.data[index]
        x = torch.FloatTensor(np.array(x)) # W, H, C
        x = x.permute(2, 0, 1) # C, W, H
        y = torch.zeros((10, ))
        y[y_idx] = 1.0
        return x, y

    def __len__(self):
        return len(self.data)


dataloader_train = torch.utils.data.DataLoader(
    dataset=DatasetCIFAR10(is_train=True),
    batch_size=BATCH_SIZE,
    shuffle=True
)
dataloader_test = torch.utils.data.DataLoader(
    dataset=DatasetCIFAR10(is_train=False),
    batch_size=BATCH_SIZE,
    shuffle=False
)

import torch.nn.functional

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torchvision.models.densenet121(pretrained=True).features
        self.fc = torch.nn.Linear(
            in_features=self.encoder.norm5.num_features, #1024
            out_features=10
        )

    def forward(self, x):
        # x = (B, 3, 32, 32)
        out = self.encoder.forward(x) # x = (B, 1024, 4, 4)
        out = torch.nn.functional.adaptive_avg_pool2d(out, output_size=(1,1)) # x = (B, 1024, 1, 1)
        out = out.view(out.size(0), -1)
        out = self.fc.forward(out)
        out = torch.nn.functional.softmax(out, dim=1)
        return out

HIDDEN_SIZE = 64
IMAGE_SIZE = 32
PATCH_SIZE = 2
MSA_HEADS = 4

class ModelVisionTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encode_positional = torch.nn.Embedding(
            num_embeddings=int((IMAGE_SIZE/PATCH_SIZE)**2),
            embedding_dim=HIDDEN_SIZE
        )
        self.encode_class = torch.nn.Parameter(
            torch.rand((HIDDEN_SIZE,))
        )
        self.encode_project = torch.nn.Linear(
            in_features=int(PATCH_SIZE**2),
            out_features=HIDDEN_SIZE
        )

        self.layer_norm_1 = torch.nn.LayerNorm(
            normalized_shape=[HIDDEN_SIZE]
        )

        self.msa_1 = torch.nn.MultiheadAttention(
            embed_dim=HIDDEN_SIZE,
            num_heads=MSA_HEADS
        )

        self.layer_norm_2 = torch.nn.LayerNorm(
            normalized_shape=[HIDDEN_SIZE]
        )

        self.mlp_2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE),
        )

        self.layer_norm_3 = torch.nn.LayerNorm(
            normalized_shape=[HIDDEN_SIZE]
        )

        self.msa_3 = torch.nn.MultiheadAttention(
            embed_dim=HIDDEN_SIZE,
            num_heads=MSA_HEADS
        )

        self.layer_norm_4 = torch.nn.LayerNorm(
            normalized_shape=[HIDDEN_SIZE]
        )

        self.mlp_4 = torch.nn.Sequential(
            torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE),
        )

        self.layer_norm_5 = torch.nn.LayerNorm(
            normalized_shape=[HIDDEN_SIZE]
        )

    def forward(self, x):
        # x = (B, 3, 32, 32)
        out = torch.zeros((BATCH_SIZE, 10))
        #TODO finish

        return out

model = ModelVisionTransformer()
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
dummy = torch.rand(size=(BATCH_SIZE, 3, 32, 32))
out_dummy = model.forward(dummy)

for epoch in range(1, 10):
    for data_loader in [dataloader_train, dataloader_test]:
        losses = []
        mode = 'test'
        for x, y in data_loader:
            y_prim = model.forward(x)
            loss = torch.mean(-y*torch.log(y_prim)) # .value .grad +=  .grad_fn

            losses.append(loss.item()) # y_prim.data.numpy()

            if data_loader == dataloader_train:
                mode = 'train'
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            if len(losses) > 2: #HACK
                break
        print(f'epoch: {epoch} mode: {mode} loss: {np.mean(losses)}')