import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchattacks
from tqdm import tqdm
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# --- Hyperparameters ---
EPOCHS = 120
LEARNING_RATE = 0.1
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
LOSS_BETA = 0.25
LOSS_GAMMA = 6.0
ADV_EPSILON = 8/255
ADV_ALPHA = 2/255
ADV_STEPS = 10
BATCH_SIZE = 128  # This will be per-GPU
NUM_WORKERS = 4
PIN_MEMORY = True
CHECKPOINT_DIR = 'apc_project/checkpoints'

def setup_ddp():
    """Initializes the distributed process group."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

# --- Model Architectures ---

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class PurifierUNet(nn.Module):
    def __init__(self):
        super(PurifierUNet, self).__init__()
        self.enc1_1 = ConvBlock(3, 64)
        self.enc1_2 = ConvBlock(64, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2_1 = ConvBlock(64, 128)
        self.enc2_2 = ConvBlock(128, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3_1 = ConvBlock(128, 256)
        self.enc3_2 = ConvBlock(256, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2_1 = ConvBlock(256, 128)
        self.dec2_2 = ConvBlock(128, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1_1 = ConvBlock(128, 64)
        self.dec1_2 = ConvBlock(64, 64)

        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1_2(self.enc1_1(x))
        enc2 = self.enc2_2(self.enc2_1(self.pool1(enc1)))
        enc3 = self.enc3_2(self.enc3_1(self.pool2(enc2)))

        dec2 = self.upconv2(enc3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2_2(self.dec2_1(dec2))

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1_2(self.dec1_1(dec1))

        return self.final_conv(dec1)

class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out

class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1, padding=1, bias=True)
        self.layer1 = self._wide_layer(WideBasic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

def WideResNet34_10(num_classes=10, dropout_rate=0.3):
    return WideResNet(34, 10, dropout_rate, num_classes)

class ComposedModel(nn.Module):
    def __init__(self, purifier, classifier):
        super().__init__()
        self.purifier = purifier
        self.classifier = classifier

    def forward(self, x):
        return self.classifier(self.purifier(x))

def main(rank, world_size):
    print(f"==> Starting process {rank}/{world_size}..")
    # Data
    print(f"==> [{rank}] Preparing data..")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, sampler=train_sampler)

    # Models
    print(f"==> [{rank}] Building models..")
    purifier = PurifierUNet().to(rank)
    classifier = WideResNet34_10().to(rank)
    purifier = DDP(purifier, device_ids=[rank])
    classifier = DDP(classifier, device_ids=[rank])

    # Optimizer and Scheduler
    optimizer = optim.SGD(
        list(purifier.parameters()) + list(classifier.parameters()),
        lr=LEARNING_RATE * world_size,  # Scale learning rate
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training loop
    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch) # Important for shuffling
        if rank == 0:
            print(f"\nEpoch: {epoch+1}/{EPOCHS}")
        purifier.train()
        classifier.train()
        
        train_loss = 0
        train_loss_ce_acc = 0
        train_loss_recon_acc = 0
        train_loss_robust_acc = 0
        correct = 0
        total = 0

        progress_bar = tqdm(trainloader, desc=f'Epoch {epoch+1}', disable=(rank != 0))

        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(rank), labels.to(rank)

            # Note: DDP handles model wrapping, no need to re-wrap in loop
            composed_model = ComposedModel(purifier.module, classifier.module)
            atk = torchattacks.PGD(composed_model, eps=ADV_EPSILON, alpha=ADV_ALPHA, steps=ADV_STEPS)

            adv_images = atk(images, labels)

            optimizer.zero_grad()

            purified_clean = purifier(images)
            purified_adv = purifier(adv_images)
            logits_clean = classifier(purified_clean)
            logits_adv = classifier(purified_adv)

            loss_ce = F.cross_entropy(logits_clean, labels)
            loss_recon = F.l1_loss(purified_adv, images)
            loss_robust = F.kl_div(
                F.log_softmax(logits_adv, dim=1),
                F.softmax(logits_clean.detach(), dim=1),
                reduction='batchmean'
            )

            total_loss = loss_ce + LOSS_BETA * loss_recon + LOSS_GAMMA * loss_robust

            total_loss.backward()
            optimizer.step()

            # Metrics are only tracked on rank 0 for simplicity
            if rank == 0:
                train_loss += total_loss.item()
                train_loss_ce_acc += loss_ce.item()
                train_loss_recon_acc += loss_recon.item()
                train_loss_robust_acc += loss_robust.item()
                _, predicted = logits_clean.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                progress_bar.set_postfix({
                    'Loss': f'{train_loss/(batch_idx+1):.3f}',
                    'CE': f'{train_loss_ce_acc/(batch_idx+1):.3f}',
                    'Recon': f'{train_loss_recon_acc/(batch_idx+1):.3f}',
                    'Robust': f'{train_loss_robust_acc/(batch_idx+1):.3f}',
                    'Acc': f'{100.*correct/total:.3f}% ({correct}/{total})'
                })

        scheduler.step()

        if rank == 0 and (epoch + 1) % 20 == 0:
            print("==> Saving checkpoint..")
            torch.save(purifier.module.state_dict(), os.path.join(CHECKPOINT_DIR, f'purifier_epoch_{epoch+1}.pth'))
            torch.save(classifier.module.state_dict(), os.path.join(CHECKPOINT_DIR, f'classifier_epoch_{epoch+1}.pth'))

    if rank == 0:
        print("==> Saving final models..")
        torch.save(purifier.module.state_dict(), os.path.join(CHECKPOINT_DIR, 'purifier_final.pth'))
        torch.save(classifier.module.state_dict(), os.path.join(CHECKPOINT_DIR, 'classifier_final.pth'))
        print("Training finished.")

if __name__ == '__main__':
    setup_ddp()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if rank == 0 and not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    main(rank, world_size)
    cleanup_ddp()
