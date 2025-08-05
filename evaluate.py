import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time
from autoattack import AutoAttack
import os

# Import model architectures from train_apc.py
from train_apc import PurifierUNet, WideResNet34_10, ComposedModel

# --- Setup ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128
ADV_EPSILON = 8/255
CHECKPOINT_DIR = 'apc_project/checkpoints'

def main():
    # Load Models
    print("==> Loading models..")
    purifier = PurifierUNet().to(DEVICE)
    classifier = WideResNet34_10().to(DEVICE)

    purifier_path = os.path.join(CHECKPOINT_DIR, 'purifier_final.pth')
    classifier_path = os.path.join(CHECKPOINT_DIR, 'classifier_final.pth')

    purifier.load_state_dict(torch.load(purifier_path, map_location=DEVICE))
    classifier.load_state_dict(torch.load(classifier_path, map_location=DEVICE))

    purifier.eval()
    classifier.eval()

    composed_model = ComposedModel(purifier, classifier).to(DEVICE)
    composed_model.eval()

    # Load Data
    print("==> Preparing data..")
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # --- Clean Accuracy Evaluation ---
    print("\n==> Evaluating Clean Accuracy..")
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = composed_model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    clean_accuracy = 100. * correct / total
    print(f"Clean Accuracy: {clean_accuracy:.2f}%")

    # --- Robust Accuracy Evaluation (AutoAttack) ---
    print("\n==> Evaluating Robust Accuracy with AutoAttack..")
    x_test = torch.cat([x for x, y in testset], dim=0)
    y_test = torch.tensor([y for x, y in testset])

    adversary = AutoAttack(composed_model, norm='Linf', eps=ADV_EPSILON, version='standard', verbose=True)
    adversary.run_standard_evaluation(x_test.to(DEVICE), y_test.to(DEVICE), bs=BATCH_SIZE)

    # --- Inference Speed Test ---
    print("\n==> Testing Inference Speed..")
    dummy_input = torch.randn(BATCH_SIZE, 3, 32, 32).to(DEVICE)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 100
    timings = torch.zeros((repetitions, 1))

    # Warm-up
    for _ in range(10):
        _ = composed_model(dummy_input)

    # Measurement
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = composed_model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    avg_batch_time = timings.mean().item()
    avg_image_time = avg_batch_time / BATCH_SIZE
    print(f"Average time per image: {avg_image_time:.6f} ms")

if __name__ == '__main__':
    main()