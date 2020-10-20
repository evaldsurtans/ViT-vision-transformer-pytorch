import torch
import torch.utils.data
import torchvision
import numpy as np
import torch.nn.functional
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20, 5)

# normally pass hyper params through argparse
BATCH_SIZE = 128
HIDDEN_SIZE = 512
PATCH_SIZE = 4
MSA_HEADS = 8
TRANSFORMER_LAYERS = 12
LEARNING_RATE = 1e-3
EPOCHS = 100

CHANNELS = 1
IMAGE_SIZE = 28
OUTPUT_CLASSES = 10
DEVICE = 'cpu'

IS_AUTO_ENCODER_HACK = False # for better gradient flow learn something to predict on other seq steps
COEF_AE = 1e-2
IS_MEAN_FEATURES_HACK = False # for better gradient flow use all outputs

if torch.cuda.is_available():
    DEVICE = 'cuda'

class DatasetMNIST(torch.utils.data.Dataset):

    def __init__(self, is_train):
        super().__init__()

        self.data = torchvision.datasets.mnist.MNIST(
            root='./tmp',
            train=is_train,
            download=True
        )

    def __getitem__(self, index):
        x, y_idx = self.data[index]
        x = torch.FloatTensor(np.array(x)) # W, H
        x = torch.unsqueeze(x, dim=0) # C, W, H

        # for CIFAR10
        #x = x.permute(2, 0, 1) # C, W, H

        y = torch.zeros((10, ))
        y[y_idx] = 1.0
        return x, y

    def __len__(self):
        return len(self.data)


dataloader_train = torch.utils.data.DataLoader(
    dataset=DatasetMNIST(is_train=True),
    batch_size=BATCH_SIZE,
    shuffle=True
)
dataloader_test = torch.utils.data.DataLoader(
    dataset=DatasetMNIST(is_train=False),
    batch_size=BATCH_SIZE,
    shuffle=False
)


class PositionalEncoding(torch.nn.Module):

    def __init__(self, num_embeddings, embedding_dim):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(num_embeddings, embedding_dim)
        position = torch.arange(0, num_embeddings, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-np.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, idxes):
        return self.pe[idxes, :]

class ModelClassic(torch.nn.Module):
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


class TransformerLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_norm_1 = torch.nn.LayerNorm(
            normalized_shape=HIDDEN_SIZE
        )

        self.msa_1 = torch.nn.MultiheadAttention(
            embed_dim=HIDDEN_SIZE,
            num_heads=MSA_HEADS,
            bias=False,
            add_bias_kv=False
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

    def forward(self, z_0):

        z_1 = self.layer_norm_1.forward(z_0)

        z_1_msa, z_1_attn = self.msa_1.forward(
            query=z_1,
            key=z_1,
            value=z_1
        )

        z_1_msa_skip = z_1_msa + z_0

        z_2 = self.layer_norm_2.forward(z_1_msa_skip)
        z_2_mlp = self.mlp_2.forward(z_2)

        z_2_mlp_skip = z_2_mlp + z_1_msa_skip

        return z_2_mlp_skip


class ModelVisionTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # torch.nn.Embedding
        self.encode_positional = PositionalEncoding(
            num_embeddings=int((IMAGE_SIZE/PATCH_SIZE)**2) + 1, # For class token extra embedding
            embedding_dim=HIDDEN_SIZE
        )
        self.encode_class = torch.nn.Parameter(
            torch.normal(mean=0.0, std=1.0, size=(HIDDEN_SIZE,))
        )

        self.encode_project = torch.nn.Linear(
            in_features=int(CHANNELS * PATCH_SIZE**2),
            out_features=HIDDEN_SIZE
        )

        self.transformers = torch.nn.Sequential(
            *[TransformerLayer() for _ in range(TRANSFORMER_LAYERS)]
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
        # x = (B, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)

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

        z_4_mlp_skip = self.transformers.forward(z_0)

        z_5 = self.layer_norm_5.forward(z_4_mlp_skip)

        z_5_class_token = z_5[:, 0] # drop all pixel encodings, care only about the class token

        if IS_MEAN_FEATURES_HACK:
            z_5_class_token = torch.mean(z_5, dim=1)

        y_prim = self.mlp_5_head.forward(z_5_class_token)

        y_prim_softmax = torch.nn.functional.softmax(y_prim, dim=1)

        if IS_AUTO_ENCODER_HACK:
            return y_prim_softmax, z_5[:, 1:], x_projected
        return y_prim_softmax


model = ModelVisionTransformer()
optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=LEARNING_RATE
)

model = model.to(DEVICE)

dummy = torch.rand(size=(BATCH_SIZE, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)).to(DEVICE)
out_dummy = model.forward(dummy)

print('starting training')

metrics = { }
for mode in ['train', 'test']:
    metrics[f'{mode}_losses'] = []
    metrics[f'{mode}_acc'] = []

# for AE map to random order to encaurage attention to work
if IS_AUTO_ENCODER_HACK:
    random_patch_order = torch.LongTensor(np.random.permutation(int((IMAGE_SIZE/PATCH_SIZE)**2)))
    random_patch_order = random_patch_order.to(DEVICE)

for epoch in range(1, EPOCHS):
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
            if IS_AUTO_ENCODER_HACK:
                y_prim, y_prim_pathces, y_patches = model.forward(x)
            else:
                y_prim = model.forward(x)
            loss = torch.mean(-y*torch.log(y_prim)) # .value .grad +=  .grad_fn

            if IS_AUTO_ENCODER_HACK:
                loss += COEF_AE * torch.mean((y_prim_pathces[:, random_patch_order] - y_patches) ** 2)

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
    plt.plot(metrics['train_losses'], 'r-', label='train_loss')
    plt.plot(metrics['test_losses'], 'b-', label='test_loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(metrics['train_acc'], 'r-', label='train_acc')
    plt.plot(metrics['test_acc'], 'b-', label='test_acc')
    plt.legend()

    plt.show()
