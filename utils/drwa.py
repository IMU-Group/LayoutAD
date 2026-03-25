import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_metrics(csv_path: str, save_dir: str):

    plt.style.use("seaborn-v0_8")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 14

    if not os.path.exists(csv_path):
        print(f"[Warning] CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["i-auroc"], marker="o", label="Instance AUROC")
    plt.plot(df["epoch"], df["p-auroc"], marker="s", label="Full Pixel AUROC")
    plt.plot(df["epoch"], df["aupro"], marker="^", label="Anomaly Pixel AUROC")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("AUROC", fontsize=12)
    plt.title("AUROC over Epochs", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "auroc_curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    # plt.plot(df["epoch"], df["f1"], marker="o", color="green", label="F1 Score")
    plt.plot(df["epoch"], df["fpr"], marker="x", color="red", label="FPR")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title("FPR over Epochs", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fpr_curve.png"), dpi=300)
    plt.close()

    if "train_loss" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df["loss"], marker="o", color="orange")
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Training Loss over Epochs", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "train_loss_curve.png"), dpi=300)
        plt.close()

    print(f"Metric plots saved in {save_dir}")
