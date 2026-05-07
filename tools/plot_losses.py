#!/usr/bin/env python3
"""
Training Loss Visualization Script

Reads val_stats.json from train_sam3_lora_native.py and generates loss curves.
"""

import json
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def load_stats(json_path):
    stats = []
    with open(json_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                stats.append(json.loads(line))
    return stats


def plot_losses(stats, output_path=None, show_best=True):
    epochs = [s['epoch'] for s in stats]
    train_losses = [s['train_loss'] for s in stats]
    val_losses = [s['val_loss'] for s in stats]

    # Find best epoch (lowest val_loss)
    best_idx = val_losses.index(min(val_losses))
    best_epoch = epochs[best_idx]
    best_val = val_losses[best_idx]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot curves
    ax.plot(epochs, train_losses, 'b-', linewidth=1.5, label='Train Loss', alpha=0.8)
    ax.plot(epochs, val_losses, 'r-', linewidth=1.5, label='Validation Loss', alpha=0.8)

    # Mark best epoch
    if show_best:
        ax.scatter([best_epoch], [best_val], color='green', s=100, zorder=5, marker='*')
        ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5)
        ax.annotate(
            f'Best: Epoch {best_epoch}\nVal Loss: {best_val:.4f}',
            xy=(best_epoch, best_val),
            xytext=(best_epoch + len(epochs) * 0.05, best_val),
            fontsize=10,
            color='green',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

    # Formatting
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('SAM3 LoRA Training Loss Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add final values as text
    final_train = train_losses[-1]
    final_val = val_losses[-1]
    textstr = f'Final Train Loss: {final_train:.4f}\nFinal Val Loss: {final_val:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def plot_losses_dual_axis(stats, output_path=None):
    epochs = [s['epoch'] for s in stats]
    train_losses = [s['train_loss'] for s in stats]
    val_losses = [s['val_loss'] for s in stats]

    best_idx = val_losses.index(min(val_losses))
    best_epoch = epochs[best_idx]
    best_val = val_losses[best_idx]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Train loss (left axis)
    color1 = 'tab:blue'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Train Loss', color=color1, fontsize=12)
    line1 = ax1.plot(epochs, train_losses, color=color1, linewidth=1.5, label='Train Loss')[0]
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    # Val loss (right axis)
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Validation Loss', color=color2, fontsize=12)
    line2 = ax2.plot(epochs, val_losses, color=color2, linewidth=1.5, label='Validation Loss')[0]
    ax2.tick_params(axis='y', labelcolor=color2)

    # Best epoch marker
    if best_epoch:
        ax2.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5)
        ax2.scatter([best_epoch], [best_val], color='green', s=100, zorder=5, marker='*')
        ax2.annotate(
            f'Best: Epoch {best_epoch}\nVal: {best_val:.4f}',
            xy=(best_epoch, best_val),
            xytext=(best_epoch + len(epochs) * 0.03, best_val * 0.9),
            fontsize=9,
            color='green',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

    # Combined legend
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=11)

    plt.title('SAM3 LoRA Training Loss Curves (Dual Axis)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize training loss curves")
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/sam3_lora/val_stats.json",
        help="Path to val_stats.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="my_loss_plot.png",
        help="Path to save plot (optional, shows plot if not specified)"
    )
    parser.add_argument(
        "--dual-axis",
        action="store_true",
        help="Use dual y-axis (different scales for train/val)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        exit(1)

    print(f"Loading stats from {input_path}...")
    stats = load_stats(input_path)
    print(f"Loaded {len(stats)} epochs")
    print(f"Best val_loss: {min(s['val_loss'] for s in stats):.4f} at epoch {min(stats, key=lambda x: x['val_loss'])['epoch']}")

    output_path = Path(args.output) if args.output else None

    if args.dual_axis:
        plot_losses_dual_axis(stats, output_path)
    else:
        plot_losses(stats, output_path)
