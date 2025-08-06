# /apc_project/evaluate.py

import torch
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
from autoattack import AutoAttack
import argparse

from models import PurifierUNet, WideResNet34_10, ComposedModel

def main():
    parser = argparse.ArgumentParser(description='APC Evaluation')
    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size')
    parser.add_argument('--purifier_path', default='./checkpoints/purifier_final.pth', type=str, help='path to purifier weights')
    parser.add_argument('--classifier_path', default='./checkpoints/classifier_final.pth', type=str, help='path to classifier weights')
    args = parser.parse_args()

    # --- Setup ---
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    ADV_EPSILON = 8 / 255
    print(f"Using device: {DEVICE}")

    # --- Data Handling ---
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # --- Model Loading ---
    purifier = PurifierUNet().to(DEVICE)
    classifier = WideResNet34_10().to(DEVICE)

    purifier.load_state_dict(torch.load(args.purifier_path, map_location=DEVICE))
    classifier.load_state_dict(torch.load(args.classifier_path, map_location=DEVICE))

    purifier.eval()
    classifier.eval()
    
    # Note: For evaluation, we do not wrap with DataParallel. 
    # AutoAttack handles batching internally and works best on a single model instance.
    composed_model = ComposedModel(purifier, classifier)

    # --- 1. Clean Accuracy Evaluation ---
    print("\nEvaluating Clean Accuracy...")
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Clean Eval"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = composed_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    clean_acc = 100 * correct / total
    print(f'Clean Accuracy: {clean_acc:.2f} %')

    # --- 2. Robust Accuracy Evaluation (AutoAttack) ---
    print("\nEvaluating Robust Accuracy with AutoAttack... (This may take a while)")
    x_test = torch.cat([x for x, y in test_loader], dim=0)
    y_test = torch.cat([y for x, y in test_loader], dim=0)
    
    adversary = AutoAttack(composed_model, norm='Linf', eps=ADV_EPSILON, version='standard', verbose=True)
    adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size)
    
    # --- 3. Inference Speed Test ---
    print("\nEvaluating Inference Speed...")
    dummy_input = torch.randn(args.batch_size, 3, 32, 32).to(DEVICE)
    
    # Warmup
    for _ in range(10):
        _ = composed_model(dummy_input)
        
    # Measurement
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings = np.zeros((100, 1))
        
        with torch.no_grad():
            for i in range(100):
                starter.record()
                _ = composed_model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[i] = curr_time
        
        avg_batch_time = np.mean(timings)
        avg_image_time = avg_batch_time / args.batch_size
        print(f'Average Inference Time per Image: {avg_image_time:.4f} ms')
    else:
        print("CUDA not available, skipping precise timing.")

if __name__ == '__main__':
    main()