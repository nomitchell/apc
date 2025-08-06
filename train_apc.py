# /apc_project/train_apc.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchattacks
from tqdm import tqdm
import os
import argparse
from torch.cuda.amp import GradScaler, autocast

from models import PurifierUNet, WideResNet34_10, ComposedModel

def main():
    # --- Argument Parser for flexibility ---
    parser = argparse.ArgumentParser(description='APC Training')
    parser.add_argument('--epochs', default=120, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=1024, type=int, help='mini-batch size')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--checkpoint_path', default='./checkpoints', type=str, help='path to save checkpoints')
    args = parser.parse_args()

    # --- Hyperparameters ---
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    WEIGHT_DECAY = 5e-4
    MOMENTUM = 0.9
    LOSS_BETA = 0.25
    LOSS_GAMMA = 6.0
    ADV_EPSILON = 8 / 255
    ADV_ALPHA = 2 / 255
    ADV_STEPS = 10
    
    # --- Setup ---
    print(f"Using device: {DEVICE}")
    if DEVICE == 'cuda':
        torch.backends.cudnn.benchmark = True
    os.makedirs(args.checkpoint_path, exist_ok=True)

    # --- Data Handling ---
    print("==> Preparing data..")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=32, pin_memory=True)

    # --- Models, Optimizer, Scheduler ---
    print("==> Building models..")
    purifier = PurifierUNet()
    classifier = WideResNet34_10()

    # --- Multi-GPU Support (DataParallel) ---
    purifier = purifier.to(DEVICE)
    classifier = classifier.to(DEVICE)
    n_gpu = torch.cuda.device_count()
    if DEVICE == 'cuda' and n_gpu > 1:
        print(f"Using {n_gpu} GPUs!")
        purifier = nn.DataParallel(purifier)
        classifier = nn.DataParallel(classifier)

    optimizer = optim.SGD(
        list(purifier.parameters()) + list(classifier.parameters()),
        lr=args.lr,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    criterion_ce = nn.CrossEntropyLoss()
    criterion_l1 = nn.L1Loss()
    criterion_kl = nn.KLDivLoss(reduction='batchmean')

    # --- Attacker Setup ---
    composed_model = ComposedModel(purifier, classifier)
    atk = torchattacks.PGD(composed_model, eps=ADV_EPSILON, alpha=ADV_ALPHA, steps=ADV_STEPS, random_start=True)

    # --- AMP Scaler ---
    scaler = GradScaler()

    # --- Training Loop ---
    print("==> Starting Training..")
    for epoch in range(args.epochs):
        purifier.train()
        classifier.train()
        
        running_loss = 0.0
        running_robust_correct = 0
        total_train_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()

            with autocast():
                adv_images = atk(images, labels)
                purified_clean = purifier(images)
                purified_adv = purifier(adv_images)

                if i % 50 == 0:
                    torchvision.utils.save_image(images, 'debug_clean.png', normalize=True)
                    torchvision.utils.save_image(adv_images, 'debug_adversarial.png', normalize=True)
                    torchvision.utils.save_image(purified_adv, 'debug_purified_adversarial.png', normalize=True)

                logits_clean = classifier(purified_clean)
                logits_adv = classifier(purified_adv)

                loss_ce = criterion_ce(logits_clean, labels)
                loss_recon = criterion_l1(purified_adv, images)
                loss_robust = criterion_kl(
                    F.log_softmax(logits_adv, dim=1),
                    F.softmax(logits_clean.detach(), dim=1)
                )
                
                total_loss = loss_ce + LOSS_BETA * loss_recon + LOSS_GAMMA * loss_robust

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += total_loss.item()
            
            _, predicted_adv = logits_adv.max(1)
            total_train_samples += labels.size(0)
            running_robust_correct += predicted_adv.eq(labels).sum().item()

            pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'CE': f'{loss_ce.item():.4f}',
                'Recon': f'{loss_recon.item():.4f}',
                'Robust': f'{loss_robust.item():.4f}'
            })

        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader)
        robust_acc = 100. * running_robust_correct / total_train_samples
        print(f"\nEpoch {epoch+1} Summary: Avg Loss: {epoch_loss:.4f}, Robust Acc: {robust_acc:.2f}%, LR: {scheduler.get_last_lr()[0]:.5f}")

        if (epoch + 1) % 20 == 0 or (epoch + 1) == args.epochs:
            print(f"Saving checkpoint at epoch {epoch+1}...")
            purifier_state = purifier.module.state_dict() if n_gpu > 1 else purifier.state_dict()
            classifier_state = classifier.module.state_dict() if n_gpu > 1 else classifier.state_dict()
            torch.save(purifier_state, os.path.join(args.checkpoint_path, f'purifier_epoch_{epoch+1}.pth'))
            torch.save(classifier_state, os.path.join(args.checkpoint_path, f'classifier_epoch_{epoch+1}.pth'))
    
    print("Training finished.")

if __name__ == '__main__':
    main()