import torch
import torch.utils.data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20, 10)

# normally pass hyper params through argparse
BATCH_SIZE = 64
HIDDEN_SIZE = 96
IMAGE_SIZE = 32
PATCH_SIZE = 2
MSA_HEADS = 4
CHANNELS = 3
OUTPUT_CLASSES = 10

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
    #torch.cuda.device_count()

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

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torchvision.models.densenet121(pretrained=True).features
        self.fc = torch.nn.Linear(
            in_features=self.encoder.norm5.num_features, #1024
            out_features=OUTPUT_CLASSES
        )

    def forward(self, x):
        # x = (B, 3, 32, 32)
        out = self.encoder.forward(x) # x = (B, 1024, 4, 4)
        out = torch.nn.functional.adaptive_avg_pool2d(out, output_size=(1,1)) # x = (B, 1024, 1, 1)
        out = out.view(out.size(0), -1)
        out = self.fc.forward(out)
        out = torch.nn.functional.softmax(out, dim=1)
        return out

class ModelVisionTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encode_positional = torch.nn.Embedding(
            num_embeddings=int((IMAGE_SIZE/PATCH_SIZE)**2) + 1, # For class token extra embedding
            embedding_dim=HIDDEN_SIZE
        )
        self.encode_class = torch.nn.Parameter(
            torch.rand((HIDDEN_SIZE,))
        )

        self.encode_project = torch.nn.Linear(
            in_features=int(CHANNELS * PATCH_SIZE**2),
            out_features=HIDDEN_SIZE
        )

        self.layer_norm_1 = torch.nn.LayerNorm(
            normalized_shape=HIDDEN_SIZE
        )

        self.msa_1 = torch.nn.MultiheadAttention(
            embed_dim=HIDDEN_SIZE,
            num_heads=MSA_HEADS
        )

        self.layer_norm_2 = torch.nn.LayerNorm(
            normalized_shape=HIDDEN_SIZE
        )

        self.mlp_2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE),
        )

        self.layer_norm_3 = torch.nn.LayerNorm(
            normalized_shape=HIDDEN_SIZE
        )

        self.msa_3 = torch.nn.MultiheadAttention(
            embed_dim=HIDDEN_SIZE,
            num_heads=MSA_HEADS
        )

        self.layer_norm_4 = torch.nn.LayerNorm(
            normalized_shape=HIDDEN_SIZE
        )

        self.mlp_4 = torch.nn.Sequential(
            torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE),
        )

        self.layer_norm_5 = torch.nn.LayerNorm(
            normalized_shape=HIDDEN_SIZE
        )

        self.mlp_5_head = torch.nn.Sequential(
            torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=OUTPUT_CLASSES),
        )

    def forward(self, x):
        # x = (B, 3, 32, 32)

        x_patches = torch.nn.functional.unfold(x, (PATCH_SIZE, PATCH_SIZE), dilation=1, padding=0, stride=(PATCH_SIZE, PATCH_SIZE))

        x_patches_trans = x_patches.permute((0, 2, 1))
        x_projected = self.encode_project.forward(x_patches_trans)

        # self.encode_class (HIDDEN_SIZE,)
        encode_classes_unsqueeze = self.encode_class.unsqueeze(dim=0) # (1, HIDDEN_SIZE)
        encode_classes_repeated = encode_classes_unsqueeze.expand((x.size(0), -1, -1)) # (B, 1, HIDDEN_SIZE)

        x_0 = torch.cat((
            encode_classes_repeated,
            x_projected
        ), dim=1)

        pos_indexes = torch.arange(0, x_0.size(1)).to(DEVICE) # new tensor in the heap memory

        embs_0 = self.encode_positional.forward(pos_indexes) # (SEQ, HIDDEN_SIZE)
        embs_repeated_0 = embs_0.expand((x.size(0), -1, -1)) # (B, SEQ, HIDDEN_SIZE)

        z_0 = x_0 + embs_repeated_0

        #### Transformer Layer 1

        z_1 = self.layer_norm_1.forward(z_0)
        z_1_msa, z_1_attn = self.msa_1.forward(query=z_1, key=z_1, value=z_1)

        z_1_msa_skip = z_1_msa + z_0

        z_2 = self.layer_norm_2.forward(z_1_msa_skip)
        z_2_mlp = self.mlp_2.forward(z_2)

        z_2_mlp_skip = z_2_mlp + z_1_msa_skip

        ####

        #### Transformer Layer 2

        z_3 = self.layer_norm_3.forward(z_2_mlp_skip)
        z_3_msa, z_3_attn = self.msa_3.forward(query=z_3, key=z_3, value=z_3)

        z_3_msa_skip = z_3_msa + z_3

        z_4 = self.layer_norm_4.forward(z_3_msa_skip)
        z_4_mlp = self.mlp_4.forward(z_4)

        z_4_mlp_skip = z_4_mlp + z_3_msa_skip

        ####

        z_5 = self.layer_norm_5.forward(z_4_mlp_skip)

        z_5_class_token = z_5[:, 0] # drop all pixel encodings, care only about the class token
        y_prim = self.mlp_5_head.forward(z_5_class_token)

        y_prim_softmax = torch.nn.functional.softmax(y_prim, dim=1)

        return y_prim_softmax


model = ModelVisionTransformer()
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

model = model.to(DEVICE)

# dummy = torch.rand(size=(BATCH_SIZE, 3, 32, 32))
# out_dummy = model.forward(dummy)

print('starting training')

metrics = { }
for mode in ['train', 'test']:
    metrics[f'{mode}_losses'] = []
    metrics[f'{mode}_acc'] = []

for epoch in range(1, 10):
    for data_loader in [dataloader_train, dataloader_test]:
        losses = []
        accs = []

        mode = 'test'
        if data_loader == dataloader_train:
            mode = 'train'

        for x, y in data_loader:

            y_idxes = torch.argmax(y, dim=1)

            x = x.to(DEVICE)
            y = y.to(DEVICE)
            y_prim = model.forward(x)
            loss = torch.mean(-y*torch.log(y_prim)) # .value .grad +=  .grad_fn

            y_prim_idxes = torch.argmax(y_prim.to('cpu'), dim=1)
            acc = torch.mean((y_idxes == y_prim_idxes) * 1.0)

            accs.append(acc)
            losses.append(loss.to('cpu').item()) # y_prim.data.numpy()

            if data_loader == dataloader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            if DEVICE == 'cpu':
                if len(losses) > 2: #HACK
                    break

        metrics[f'{mode}_losses'].append(np.mean(losses))
        metrics[f'{mode}_acc'].append(np.mean(accs))
        print(f"epoch: {epoch} mode: {mode} loss: {metrics[f'{mode}_losses'][-1]} acc: {metrics[f'{mode}_acc'][-1]}")

    plt.subplot(1, 2, 1)
    plt.plot(metrics[f'train_losses'], 'r-', label='train_losses')
    plt.plot(metrics[f'test_losses'], 'b-', label='test_losses')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(metrics[f'train_acc'], 'r-', label='train_acc')
    plt.plot(metrics[f'test_acc'], 'b-', label='test_acc')
    plt.legend()

    plt.show()
