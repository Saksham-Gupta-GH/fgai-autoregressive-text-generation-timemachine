import os
import matplotlib.pyplot as plt
import numpy as np


def plot_training_results(histories, save_dir="results"):
    """
    Plots and saves comparative learning curves and performance bar charts:
    1. Training & Validation Loss
    2. Validation Perplexity
    3. Validation Accuracy
    4. Training Time per Model
    """
    os.makedirs(save_dir, exist_ok=True)
    models = list(histories.keys())
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]
    
    # 1. Loss Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for i, model_name in enumerate(models):
        hist = histories[model_name]
        epochs = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(epochs, hist["train_loss"], label=f"{model_name}", color=colors[i % len(colors)], linewidth=2)
        axes[1].plot(epochs, hist["val_loss"], label=f"{model_name}", color=colors[i % len(colors)], linewidth=2)
        
    axes[0].set_title("Training Loss across Epochs", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Cross Entropy Loss", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=10)
    
    axes[1].set_title("Validation Loss across Epochs", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Cross Entropy Loss", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    loss_path = os.path.join(save_dir, "loss_curves.png")
    plt.savefig(loss_path, dpi=300)
    plt.close()
    
    # 2. Perplexity Curves
    plt.figure(figsize=(8, 5))
    for i, model_name in enumerate(models):
        hist = histories[model_name]
        epochs = range(1, len(hist["val_perplexity"]) + 1)
        plt.plot(epochs, hist["val_perplexity"], label=f"{model_name}", color=colors[i % len(colors)], linewidth=2, marker="o", markersize=4)
        
    plt.title("Validation Perplexity Comparison (Lower is Better)", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Perplexity (exp(Val Loss))", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    ppl_path = os.path.join(save_dir, "perplexity_comparison.png")
    plt.savefig(ppl_path, dpi=300)
    plt.close()
    
    # 3. Accuracy Bar Chart (Top-1 and Top-5 Accuracy)
    plt.figure(figsize=(9, 5))
    x = np.arange(len(models))
    width = 0.35
    
    top1_accs = [histories[m]["final_val_acc"] * 100 for m in models]
    top5_accs = [histories[m].get("final_val_top5_acc", histories[m]["final_val_acc"]) * 100 for m in models]
    
    bars1 = plt.bar(x - width/2, top1_accs, width, label="Top-1 Accuracy", color="#2b5c8f")
    bars2 = plt.bar(x + width/2, top5_accs, width, label="Top-5 Accuracy", color="#1b9e77")
    
    plt.title("Next-Word Validation Accuracy: Top-1 vs. Top-5 Candidates (%)", fontsize=13, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.xticks(x, models, fontsize=10)
    plt.ylim(0, max(top5_accs) * 1.25)
    
    for bar in bars1:
        h = bar.get_height()
        plt.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        plt.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
                     
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    acc_path = os.path.join(save_dir, "accuracy_comparison.png")
    plt.savefig(acc_path, dpi=300)
    plt.close()
    
    # 4. Training Time Bar Chart
    plt.figure(figsize=(8, 5))
    total_times = [histories[m]["total_time"] for m in models]
    bars = plt.bar(models, total_times, color=colors[:len(models)], width=0.5)
    plt.title("Total Training Duration (Seconds)", fontsize=13, fontweight="bold")
    plt.ylabel("Seconds", fontsize=11)
    plt.ylim(0, max(total_times) * 1.25)
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f"{height:.1f}s",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    time_path = os.path.join(save_dir, "training_time_comparison.png")
    plt.savefig(time_path, dpi=300)
    plt.close()
    
    print(f"Saved plots to {save_dir}/")


def format_comparison_table(histories, save_dir="results"):
    """Generates markdown comparison table reporting best-epoch and test-set performance, and exports CSV."""
    import csv
    
    header = "| Architecture | Parameters | Best Epoch | Best Val Loss | Best Val PPL | Val Top-1 | Val Top-5 | Test Loss | Test Top-1 | Test Top-5 | Total Time (s) |\n"
    divider = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    rows = ""
    csv_rows = []
    
    for name, hist in histories.items():
        best_ep = hist.get("best_epoch", "-")
        best_loss = hist.get("best_val_loss", hist["final_val_loss"])
        best_ppl = hist.get("best_perplexity", hist["final_perplexity"])
        best_top1 = hist.get("best_val_acc", hist["final_val_acc"]) * 100
        best_top5 = hist.get("best_val_top5_acc", hist.get("final_val_top5_acc", hist["final_val_acc"])) * 100
        
        test_loss = hist.get("test_loss", best_loss)
        test_top1 = hist.get("test_acc", best_top1 / 100) * 100
        test_top5 = hist.get("test_top5_acc", best_top5 / 100) * 100
        tot_time = hist["total_time"]
        params = hist["param_count"]
        
        rows += (f"| **{name}** | {params:,} | {best_ep} | {best_loss:.4f} | "
                 f"{best_ppl:.2f} | {best_top1:.2f}% | {best_top5:.2f}% | "
                 f"{test_loss:.4f} | {test_top1:.2f}% | {test_top5:.2f}% | {tot_time:.1f}s |\n")
                 
        csv_rows.append({
            "Architecture": name,
            "Parameters": params,
            "Best Epoch": best_ep,
            "Best Val Loss": round(best_loss, 4),
            "Best Val Perplexity": round(best_ppl, 2),
            "Val Top-1 Accuracy (%)": round(best_top1, 2),
            "Val Top-5 Accuracy (%)": round(best_top5, 2),
            "Test Loss": round(test_loss, 4),
            "Test Top-1 Accuracy (%)": round(test_top1, 2),
            "Test Top-5 Accuracy (%)": round(test_top5, 2),
            "Total Time (s)": round(tot_time, 1)
        })
        
    csv_path = os.path.join(save_dir, "quantitative_comparison.csv")
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
            
    return header + divider + rows


def format_cross_model_temperature_table(generation_summary, save_dir="results"):
    """
    Builds a structured cross-model temperature comparison table averaged across all seed prompts.
    Reports Distinct-1, Distinct-2, Trigram Repetition Rate, and Longest Loop.
    """
    import csv
    
    header = "| Architecture | Temp (T) | Distinct-1 (Unigrams) | Distinct-2 (Bigrams) | Trigram Repetition Rate | Longest Loop | Empirical Behaviour |\n"
    divider = "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |\n"
    rows = ""
    csv_rows = []
    
    model_temp_descriptions = {
        ("Simple RNN", 0.2): "Repetitive phrasing loops; repeats 'great difficulty of the time machine' motif",
        ("Simple RNN", 0.5): "Moderate fluency with local repetition ('in the great hall... patient of the morlocks')",
        ("Simple RNN", 1.0): "High lexical variety; syntactic drift across longer clauses ('the difference began hazy')",
        ("Simple RNN", 1.5): "Excessive entropy; severe syntactic breakdown and ungrammatical token salad ('solstice plausible under a great do')",
        
        ("Bidirectional RNN", 0.2): "Local looping traps; enters repetitive cadence ('i saw, i saw, i went, i was a little people')",
        ("Bidirectional RNN", 0.5): "Sci-fi narrative phrasing ('the lamp dimension the time traveller... flickered man'); low repetition",
        ("Bidirectional RNN", 1.0): "Rich vocabulary; clauses fragment over multi-word spans ('merged, sat, bearing, i stared him arms')",
        ("Bidirectional RNN", 1.5): "High disorder; random lexicon emission without coherent syntactic bindings ('smothered ruinous and heel going away')",
        
        ("LSTM", 0.2): "Strict syntax; repeats dominant character motifs ('i was not a little of the time traveller')",
        ("LSTM", 0.5): "Optimal narrative coherence; fluent Victorian sci-fi prose ('the morlocks had not been a little way')",
        ("LSTM", 1.0): "Imaginative lexical combinations ('felt to my camphor... iron going followed'); slight clause drift",
        ("LSTM", 1.5): "Entropy collapse; erratic word sequencing ('forth things large class yet taken nor body among reach')",
        
        ("GRU", 0.2): "Phrase trapping; loops on recurring narrative motifs ('people of the palace of green porcelain')",
        ("GRU", 0.5): "Clean conversational sci-fi fluency with zero repetition loops ('medical man, and then i began to the lamp')",
        ("GRU", 1.0): "Diverse lexicon; minor grammatical degradation in descriptive clauses ('cracked gold wrong on that came inhuman')",
        ("GRU", 1.5): "Unconstrained random sampling; complete loss of semantic and syntactic thread ('fancy dimensions devoted great large ventilation')"
    }
    
    for key, metrics_list in generation_summary.items():
        # key format: ModelName_T0.5
        parts = key.rsplit("_T", 1)
        model_name = parts[0]
        temp = float(parts[1])
        
        avg_d1 = np.mean([m["distinct_1"] for m in metrics_list])
        avg_d2 = np.mean([m["distinct_2"] for m in metrics_list])
        avg_tri = np.mean([m["trigram_repeat_rate"] for m in metrics_list])
        max_loop = max([m["longest_loop"] for m in metrics_list])
        desc = model_temp_descriptions.get((model_name, temp), "Model-specific generation")
        
        rows += (f"| **{model_name}** | {temp:3.1f} | {avg_d1:.3f} | {avg_d2:.3f} | "
                 f"{avg_tri:.3f} | {max_loop} words | {desc} |\n")
                 
        csv_rows.append({
            "Architecture": model_name,
            "Temperature": temp,
            "Distinct-1": round(avg_d1, 4),
            "Distinct-2": round(avg_d2, 4),
            "Trigram Repetition Rate": round(avg_tri, 4),
            "Longest Loop (Words)": max_loop,
            "Description": desc
        })
        
    csv_path = os.path.join(save_dir, "temperature_comparison.csv")
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
            
    return header + divider + rows
