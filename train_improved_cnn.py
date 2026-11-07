"""Experiment 4: CIFAR-10 CNN trained from scratch; no pretrained weights."""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR10

class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_channels = 3
        for channels in (32, 64, 128):
            for _ in range(2):
                layers += [nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
                           nn.BatchNorm2d(channels), nn.ReLU(inplace=True)]
                in_channels = channels
            layers += [nn.MaxPool2d(2), nn.Dropout2d(0.1)]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128*4*4, 256),
                                        nn.ReLU(inplace=True), nn.Dropout(0.3),
                                        nn.Linear(256, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


def run_experiment(data_path='./data_cifar', output_dir='./experiment4_results',
                   epochs=40, batch_size=128, seed=42):
    if epochs < 1 or batch_size < 1:
        raise ValueError('epochs and batch_size must be positive')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    print('Device:', device, flush=True)
    if device.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(), flush=True)
    else:
        print('CPU training may take substantially longer.', flush=True)
    dataset = CIFAR10(data_path, train=True, download=True)
    # Stratified 45,000 / 5,000 split of the official TRAINING data.
    labels = np.asarray(dataset.targets)
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls in range(10):
        idx = rng.permutation(np.flatnonzero(labels == cls))
        val_idx.extend(idx[:500])
        train_idx.extend(idx[500:])
    # Keep compact uint8 images on the selected device to reduce transfer overhead.
    images = torch.from_numpy(dataset.data.copy()).permute(0, 3, 1, 2).to(device)
    targets = torch.tensor(dataset.targets, device=device)
    train_idx = torch.tensor(train_idx, device=device)
    val_idx = torch.tensor(val_idx, device=device)
    mean = torch.tensor((0.4914, 0.4822, 0.4465), device=device).view(1,3,1,1)
    std = torch.tensor((0.2470, 0.2435, 0.2616), device=device).view(1,3,1,1)

    def preprocess(x, augment=False):
        x = x.float().div(255)
        if augment:
            # Per-image random crop with reflection padding of four pixels.
            x = F.pad(x, (4,4,4,4), mode='reflect')
            b = x.size(0)
            offsets = torch.randint(0, 9, (b,2), device=device)
            rows = offsets[:,0,None,None] + torch.arange(32, device=device)[None,:,None]
            cols = offsets[:,1,None,None] + torch.arange(32, device=device)[None,None,:]
            x = x.permute(0,2,3,1)[torch.arange(b,device=device)[:,None,None], rows, cols]
            x = x.permute(0,3,1,2)
            flip = torch.rand(b,1,1,1,device=device) < 0.5
            x = torch.where(flip, x.flip(-1), x)
        return ((x - mean) / std).contiguous()

    model = ImprovedCNN().to(device)
    print(model, flush=True)
    print('Train:', len(train_idx), 'Validation:', len(val_idx), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def evaluate(xdata, ydata, indices):
        model.eval()
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        for idx in indices.split(batch_size):
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                logits = model(preprocess(xdata[idx]))
                loss = criterion(logits, ydata[idx])
            loss_sum += loss.float() * idx.numel()
            correct += (logits.argmax(1) == ydata[idx]).sum()
        return loss_sum.item()/len(indices), 100*correct.item()/len(indices)

    # The assignment explicitly requests test loss and test accuracy each epoch.
    # Log them for the report; checkpoint selection uses validation data only.
    testset = CIFAR10(data_path, train=False, download=True)
    test_images = torch.from_numpy(testset.data.copy()).permute(0,3,1,2).to(device)
    test_labels = torch.tensor(testset.targets, device=device)
    test_indices = torch.arange(len(testset), device=device)
    history = []
    best_accuracy, best_loss, best_epoch = -1.0, float('inf'), 0
    checkpoint = output/'best_model.pt'
    start = time.time()
    for epoch in range(1, epochs+1):
        tick = time.time()
        model.train()
        permutation = train_idx[torch.randperm(len(train_idx), device=device)]
        train_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        lr = optimizer.param_groups[0]['lr']
        for idx in permutation.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                logits = model(preprocess(images[idx], augment=True))
                loss = criterion(logits, targets[idx])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_sum += loss.detach().float() * idx.numel()
            correct += (logits.detach().argmax(1) == targets[idx]).sum()
        val_loss, val_accuracy = evaluate(images, targets, val_idx)
        epoch_test_loss, epoch_test_accuracy = evaluate(test_images, test_labels, test_indices)
        row = dict(epoch=epoch, train_loss=train_sum.item()/len(train_idx),
                   train_accuracy=100*correct.item()/len(train_idx),
                   validation_loss=val_loss, validation_accuracy=val_accuracy,
                   test_loss=epoch_test_loss, test_accuracy=epoch_test_accuracy,
                   learning_rate=lr, seconds=time.time()-tick)
        history.append(row)
        if val_accuracy > best_accuracy or (val_accuracy == best_accuracy and val_loss < best_loss):
            best_accuracy, best_loss, best_epoch = val_accuracy, val_loss, epoch
            torch.save({'model_state':model.state_dict(), 'epoch':epoch,
                        'validation_accuracy':val_accuracy, 'seed':seed}, checkpoint)
        scheduler.step()
        (output/'history.json').write_text(json.dumps(history, indent=2))
        print(f"Epoch {epoch:02d}/{epochs} train loss {row['train_loss']:.4f} | "
              f"val loss {val_loss:.4f} | val accuracy {val_accuracy:.2f}% | "
              f"test accuracy {epoch_test_accuracy:.2f}% | {row['seconds']:.1f}s", flush=True)
    # Reload the validation-selected checkpoint and report its test performance.
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved['model_state'])
    test_loss, test_accuracy = evaluate(test_images, test_labels, test_indices)
    results = dict(best_epoch=best_epoch, best_validation_accuracy=best_accuracy,
                   selected_epoch_training_loss=history[best_epoch-1]['train_loss'],
                   final_epoch_training_loss=history[-1]['train_loss'],
                   test_loss=test_loss, test_accuracy=test_accuracy,
                   epochs=epochs, batch_size=batch_size, seed=seed,
                   train_samples=45000, validation_samples=5000, test_samples=10000,
                   torch_version=str(torch.__version__), device=str(device),
                   elapsed_seconds=time.time()-start)
    (output/'results.json').write_text(json.dumps(results, indent=2))
    print('FINAL RESULTS\n'+json.dumps(results, indent=2), flush=True)
    return history, results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='./data_cifar')
    parser.add_argument('--output-dir', default='./experiment4_results')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    run_experiment(args.data_path, args.output_dir, args.epochs, args.batch_size)
