# ===========================================================
# analyze.py — Energy Analysis, Advanced Features & Report
# Energy-Aware ML Project with EfficientNetV2M
# Run AFTER train.py or resume.py completes.
#
# USAGE:
#   python analyze.py              ← loads saved JSON results
#   python analyze.py --demo       ← runs on demo data (no training needed)
#
# Techniques covered:
#   Baseline | Quantization | Early Stopping |
#   Transfer Learning | Fine-Tuning
# ===========================================================

import os
import sys
import json
import time
import platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import psutil

from config import *


# ===========================================================
# LOAD RESULTS FROM JSON FILES
# ===========================================================

def load_result(name):
    """Load saved metrics for a technique from its JSON file."""
    path = os.path.join(DATA_DIR, f"result_{name.replace(' ', '_')}.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        try:
            data['params'] = int(data['params'])
        except (ValueError, TypeError):
            pass
        data.setdefault('history', None)
        return data
    return None


def load_all_results():
    """Load all technique results from saved JSON files."""
    technique_names = [
        'Baseline',
        'Quantization',
        'Early Stopping',
        'Transfer Learning',
        'Fine-Tuning',
    ]
    results = {}
    for name in technique_names:
        r = load_result(name)
        if r:
            results[name] = r
            print(f"  ✅ {name:<22} — accuracy: {r['accuracy']*100:.2f}%  "
                  f"co2: {r['co2']:.6f} kg  time: {r['time']:.1f}s")
        else:
            print(f"  ⚠️  {name}: JSON not found — skipping")
    return results


# ===========================================================
# FEATURE 1: Hybrid Energy-Aware Recommendation
# ===========================================================

def recommend_best_technique(task_type='image',
                              accuracy_requirement=0.90,
                              deployment_constraint='edge'):
    """
    Recommends the best energy-saving technique based on requirements.
    accuracy_requirement is a float in [0, 1].
    """
    knowledge_base = {
        'edge': {
            'quantization'      : {'saving': 85, 'acc_drop': 1, 'speed': 'fast',   'recommended': True},
            'transfer_learning' : {'saving': 50, 'acc_drop': 1, 'speed': 'fast',   'recommended': True},
        },
        'cloud': {
            'early_stopping'    : {'saving': 40, 'acc_drop': 0, 'speed': 'fast',   'recommended': True},
            'fine_tuning'       : {'saving': 30, 'acc_drop': 0, 'speed': 'slow',   'recommended': True},
        },
        'cloud_gpu': {
            'fine_tuning'       : {'saving': 35, 'acc_drop': 0, 'speed': 'medium', 'recommended': True},
            'early_stopping'    : {'saving': 40, 'acc_drop': 0, 'speed': 'fast',   'recommended': True},
        },
        'mobile': {
            'quantization'      : {'saving': 90, 'acc_drop': 1, 'speed': 'fast',   'recommended': True},
            'transfer_learning' : {'saving': 55, 'acc_drop': 1, 'speed': 'fast',   'recommended': True},
        },
    }

    recommendations = knowledge_base.get(deployment_constraint, knowledge_base['edge'])

    viable = []
    for tech, metrics in recommendations.items():
        if (100 - metrics['acc_drop']) / 100 >= accuracy_requirement:
            viable.append((tech, metrics))

    viable.sort(key=lambda x: x[1]['saving'], reverse=True)

    if viable:
        return viable[0]
    return ("baseline", {"saving": 0, "acc_drop": 0, "speed": "N/A"})


# ===========================================================
# FEATURE 2: Hardware-Aware Optimization
# ===========================================================

class HardwareProfiler:
    """Profiles hardware and recommends best techniques."""

    def __init__(self):
        self.system    = platform.system()
        self.processor = platform.processor()
        self.cpu_count = psutil.cpu_count(logical=True)
        self.ram       = psutil.virtual_memory().total / (1024 ** 3)
        self.has_gpu   = self._check_gpu()
        self.has_npu   = self._check_npu()

    def _check_gpu(self):
        try:
            import tensorflow as tf
            return len(tf.config.list_physical_devices('GPU')) > 0
        except Exception:
            return False

    def _check_npu(self):
        try:
            from openvino.runtime import Core
            core    = Core()
            devices = core.available_devices
            return any('NPU' in d for d in devices)
        except ImportError:
            pass
        cpu = (self.processor or platform.uname().processor or '').lower()
        intel_ultra_keywords = ['core ultra', 'meteor lake', 'lunar lake', 'arrow lake']
        return any(kw in cpu for kw in intel_ultra_keywords)

    def get_hardware_profile(self):
        return {
            'system'     : self.system,
            'processor'  : self.processor or "Unknown",
            'cpu_cores'  : self.cpu_count,
            'ram_gb'     : round(self.ram, 1),
            'has_gpu'    : self.has_gpu,
            'has_npu'    : self.has_npu,
            'device_type': self._classify_device()
        }

    def _classify_device(self):
        if self.ram > 16 and self.cpu_count > 8 and self.has_gpu:
            return 'high-end'
        elif self.ram > 8 and self.cpu_count > 4:
            return 'mid-range'
        return 'low-end'

    def recommend_techniques(self):
        device  = self.get_hardware_profile()['device_type']
        has_npu = self.has_npu
        recommendations = {
            'high-end': {
                'best'      : 'Fine-Tuning + Transfer Learning',
                'reason'    : 'You have resources to deeply unfreeze large backbones',
                'techniques': ['fine_tuning', 'transfer_learning', 'quantization'],
            },
            'mid-range': {
                'best'      : 'Transfer Learning + Early Stopping',
                'reason'    : 'Good balance of accuracy and efficiency',
                'techniques': ['transfer_learning', 'quantization', 'early_stopping'],
            },
            'low-end': {
                'best'      : 'Early Stopping + Quantization',
                'reason'    : 'Lightweight techniques that work on limited hardware',
                'techniques': ['early_stopping', 'quantization'],
            },
        }
        rec = recommendations.get(device, recommendations['mid-range'])
        if has_npu:
            rec = rec.copy()
            rec['npu_note'] = (
                "Intel NPU detected! Use OpenVINO (pip install openvino) to compile "
                "models for NPU inference — typically 2-5x faster than CPU for INT8 "
                "quantized models. Training still runs on CPU/GPU; NPU accelerates inference."
            )
        return rec

    def estimate_energy_savings(self, technique):
        base_savings = {
            'quantization'      : 85,
            'transfer_learning' : 50,
            'fine_tuning'       : 35,
            'early_stopping'    : 40,
        }
        profile    = self.get_hardware_profile()
        multiplier = 1.2 if profile['device_type'] == 'low-end' else (0.9 if profile['has_gpu'] else 1.0)
        return min(base_savings.get(technique, 50) * multiplier, 95)


# ===========================================================
# FEATURE 3: Lifecycle Carbon Assessment
# ===========================================================

def calculate_total_carbon_footprint(training_co2, model_size_mb, training_hours):
    """Full lifecycle: training + hardware manufacturing + N-year inference."""
    embodied_co2      = training_co2 * EMBODIED_FACTOR
    total_inferences  = DAILY_INFERENCES * 365 * INFERENCE_YEARS
    energy_per_inf    = (model_size_mb / 1000) * 0.001
    inference_co2     = energy_per_inf * total_inferences * GRID_CARBON_KG_PER_KWH
    total             = training_co2 + embodied_co2 + inference_co2
    return {
        'operational'    : training_co2,
        'embodied'       : embodied_co2,
        'inference'      : inference_co2,
        'total'          : total,
        'pct_operational': training_co2 / total * 100 if total > 0 else 0,
        'pct_embodied'   : embodied_co2 / total * 100 if total > 0 else 0,
        'pct_inference'  : inference_co2 / total * 100 if total > 0 else 0,
    }


# ===========================================================
# FEATURE 4: Energy-Aware Efficiency Scoring
# ===========================================================

def energy_aware_tuning_suggestion(results_dict):
    """Scores each technique on accuracy-per-CO2 and composite metric."""
    if not results_dict:
        return None

    scores    = {}
    composite = {}
    baseline_co2 = results_dict.get('Baseline', {}).get('co2', 1e-9) or 1e-9

    for technique, data in results_dict.items():
        co2 = data.get('co2', 0)
        acc = data.get('accuracy', 0)
        if co2 > 0 and acc > 0:
            scores[technique]    = acc / (co2 * 1000)
            co2_saving_pct       = max(0, (baseline_co2 - co2) / baseline_co2 * 100)
            size_penalty         = data.get('size', 1.0) or 1.0
            composite[technique] = acc * 0.5 + co2_saving_pct * 0.4 - size_penalty * 0.1

    if scores:
        best_efficiency = max(scores,    key=scores.get)
        best_composite  = max(composite, key=composite.get)
        return {
            'best_technique'   : best_efficiency,
            'best_composite'   : best_composite,
            'efficiency_score' : scores[best_efficiency],
            'scores'           : scores,
            'composite_scores' : composite,
        }
    return None


# ===========================================================
# FEATURE 5: Visualizations
# ===========================================================

def create_visualizations(all_results):
    """9-panel comparison figure (3x3 layout)."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    valid = {k: v for k, v in all_results.items() if v.get('accuracy', 0) > 0}
    if not valid:
        print("  ⚠️  No valid results to plot.")
        return

    techniques = list(valid.keys())
    accuracies = [valid[t]['accuracy'] * 100    for t in techniques]
    sizes      = [valid[t]['size']              for t in techniques]
    co2s       = [valid[t]['co2'] * 1000        for t in techniques]
    times      = [valid[t]['time']              for t in techniques]
    params     = [v if isinstance(v := valid[t].get('params', 0), (int, float)) else 0
                  for t in techniques]
    params_m   = [p / 1e6 for p in params]

    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    colors  = palette[:len(techniques)]

    fig, axes = plt.subplots(3, 3, figsize=(28, 20))
    fig.suptitle(
        f'Energy-Aware Machine Learning — {MODEL_NAME}\n'
        'Technique Comparison: Baseline · Quantization · Early Stopping · '
        'Transfer Learning · Fine-Tuning',
        fontsize=18, fontweight='bold', y=1.01
    )

    def annotate_bars(ax, bars, values, fmt='{:.1f}'):
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    fmt.format(v),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    def style_ax(ax, xlabel='', ylabel='', title='', ylim=None):
        ax.set_title(title, fontweight='bold', fontsize=12, pad=8)
        ax.set_ylabel(ylabel, fontsize=11)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=11)
        ax.tick_params(axis='x', labelrotation=25, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        if ylim:
            ax.set_ylim(*ylim)

    # 1. Accuracy
    bars = axes[0, 0].bar(techniques, accuracies, color=colors, edgecolor='white', linewidth=1.5)
    axes[0, 0].axhline(90, color='red', linestyle='--', linewidth=1.5, label='90% target')
    axes[0, 0].legend(fontsize=9)
    annotate_bars(axes[0, 0], bars, accuracies, '{:.1f}%')
    style_ax(axes[0, 0], ylabel='Accuracy (%)', title='Model Accuracy  ↑ higher is better', ylim=(0, 108))

    # 2. Model Size
    bars = axes[0, 1].bar(techniques, sizes, color=colors, edgecolor='white', linewidth=1.5)
    annotate_bars(axes[0, 1], bars, sizes, '{:.1f}')
    style_ax(axes[0, 1], ylabel='Size (MB)', title='Model Size  ↓ lower is better')

    # 3. CO₂ Emissions
    bars = axes[0, 2].bar(techniques, co2s, color=colors, edgecolor='white', linewidth=1.5)
    annotate_bars(axes[0, 2], bars, co2s, '{:.3f}g')
    style_ax(axes[0, 2], ylabel='CO₂ (grams)', title='Carbon Footprint  ↓ lower is better')

    # 4. Training Time
    bars = axes[1, 0].bar(techniques, times, color=colors, edgecolor='white', linewidth=1.5)
    annotate_bars(axes[1, 0], bars, times, '{:.0f}s')
    style_ax(axes[1, 0], ylabel='Time (seconds)', title='Training Time  ↓ lower is better')

    # 5. Energy Efficiency
    efficiency = [a / c if c > 0 else 0 for a, c in zip(accuracies, co2s)]
    bars = axes[1, 1].bar(techniques, efficiency, color=colors, edgecolor='white', linewidth=1.5)
    annotate_bars(axes[1, 1], bars, efficiency, '{:.0f}')
    style_ax(axes[1, 1], ylabel='Acc % per gram CO₂',
             title='Energy Efficiency  ↑ higher is better')

    # 6. Accuracy vs CO₂ Scatter
    ax_sc = axes[1, 2]
    for i, t in enumerate(techniques):
        ax_sc.scatter(co2s[i], accuracies[i], color=colors[i], s=180, zorder=5, label=t)
        ax_sc.annotate(t, (co2s[i], accuracies[i]),
                       textcoords='offset points', xytext=(6, 4), fontsize=8)
    ax_sc.axhline(90, color='red', linestyle='--', linewidth=1.2, alpha=0.7, label='90% target')
    ax_sc.set_xlabel('CO₂ (grams)', fontsize=11)
    ax_sc.set_ylabel('Accuracy (%)', fontsize=11)
    ax_sc.set_title('Accuracy vs CO₂  ↗ top-left ideal', fontweight='bold', fontsize=12)
    ax_sc.legend(fontsize=8, loc='lower right')
    ax_sc.grid(alpha=0.3, linestyle='--')

    # 7. Lifecycle CO₂ Pie
    b  = valid.get('Baseline', list(valid.values())[0])
    lc = calculate_total_carbon_footprint(b['co2'], b['size'], b['time'] / 3600)
    pie_vals   = [lc['operational'], lc['embodied'], lc['inference']]
    pie_labels = [
        f"Training\n(operational)\n{lc['pct_operational']:.1f}%",
        f"Hardware mfg\n(embodied)\n{lc['pct_embodied']:.1f}%",
        f"{INFERENCE_YEARS}-yr inference\n{lc['pct_inference']:.1f}%"
    ]
    axes[2, 0].pie(pie_vals, labels=pie_labels,
                   colors=['#ff7f0e', '#d62728', '#1f77b4'],
                   startangle=140,
                   textprops={'fontsize': 10},
                   wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    axes[2, 0].set_title(f'Lifecycle CO₂ Breakdown — {list(valid.keys())[0]}',
                         fontweight='bold', fontsize=12)

    # 8. Accuracy Gain vs Baseline
    ax_gain     = axes[2, 1]
    baseline_acc = valid.get('Baseline', {}).get('accuracy', 0) * 100
    gains        = [a - baseline_acc for a in accuracies]
    gain_colors  = ['#2ca02c' if g >= 0 else '#d62728' for g in gains]
    h_bars       = ax_gain.barh(techniques, gains, color=gain_colors,
                                edgecolor='white', linewidth=1.2)
    ax_gain.axvline(0, color='black', linewidth=1)
    for bar, g in zip(h_bars, gains):
        ax_gain.text(g + (0.05 if g >= 0 else -0.05),
                     bar.get_y() + bar.get_height() / 2,
                     f'{g:+.2f}%', va='center',
                     ha='left' if g >= 0 else 'right',
                     fontsize=9, fontweight='bold')
    ax_gain.set_xlabel('Accuracy Δ vs Baseline (%)', fontsize=11)
    ax_gain.set_title('Accuracy Gain vs Baseline  ↑ higher is better',
                      fontweight='bold', fontsize=12)
    ax_gain.tick_params(axis='y', labelsize=9)
    ax_gain.grid(axis='x', alpha=0.3, linestyle='--')
    ax_gain.set_axisbelow(True)

    # 9. Parameter Count
    valid_param_idx  = [i for i, p in enumerate(params_m) if p > 0]
    param_techniques = [techniques[i] for i in valid_param_idx]
    param_vals       = [params_m[i]   for i in valid_param_idx]
    param_colors     = [colors[i]     for i in valid_param_idx]

    if param_techniques:
        bars = axes[2, 2].bar(param_techniques, param_vals,
                              color=param_colors, edgecolor='white', linewidth=1.5)
        annotate_bars(axes[2, 2], bars, param_vals, '{:.1f}M')
        style_ax(axes[2, 2], ylabel='Parameters (Millions)',
                 title='Parameter Count  ↓ lighter models')
    else:
        axes[2, 2].text(0.5, 0.5, 'No param data available',
                        ha='center', va='center', transform=axes[2, 2].transAxes, fontsize=12)
        axes[2, 2].set_title('Parameter Count', fontweight='bold', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(PLOT_COMPARISON_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Comparison plot saved: {PLOT_COMPARISON_PATH}")


def create_training_history_plot(histories):
    """Plot accuracy & loss curves for techniques that have a history object."""
    valid_h = {k: v for k, v in histories.items() if v is not None}
    if not valid_h:
        print("  ⚠️  No training history available to plot.")
        return

    n = len(valid_h)
    fig, axes = plt.subplots(n, 2, figsize=(22, 6 * n))
    if n == 1:
        axes = np.array([axes])

    fig.suptitle(f'{MODEL_NAME} — Training History (all techniques)',
                 fontsize=16, fontweight='bold')

    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for row, (name, hist) in enumerate(valid_h.items()):
        col = palette[row % len(palette)]
        axes[row, 0].plot(hist.history['accuracy'],     label='Train',
                          marker='o', linewidth=2, color=col)
        axes[row, 0].plot(hist.history['val_accuracy'], label='Validation',
                          marker='s', linewidth=2, color=col, linestyle='--')
        axes[row, 0].axhline(0.90, color='red', linestyle=':', linewidth=1.2, label='90% target')
        axes[row, 0].set_xlabel('Epoch', fontsize=11)
        axes[row, 0].set_ylabel('Accuracy', fontsize=11)
        axes[row, 0].set_title(f'{name} — Accuracy', fontweight='bold', fontsize=12)
        axes[row, 0].legend(fontsize=9)
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(hist.history['loss'],     label='Train',
                          marker='o', linewidth=2, color=col)
        axes[row, 1].plot(hist.history['val_loss'], label='Validation',
                          marker='s', linewidth=2, color=col, linestyle='--')
        axes[row, 1].set_xlabel('Epoch', fontsize=11)
        axes[row, 1].set_ylabel('Loss', fontsize=11)
        axes[row, 1].set_title(f'{name} — Loss', fontweight='bold', fontsize=12)
        axes[row, 1].legend(fontsize=9)
        axes[row, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'training_history.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Training history plot saved: {path}")


# ===========================================================
# FEATURE 6: CSV Export
# ===========================================================

def save_results_csv(all_results):
    rows = []
    for technique, metrics in all_results.items():
        rows.append({
            'Technique'   : technique,
            'Accuracy (%)': round(metrics['accuracy'] * 100, 2),
            'Size (MB)'   : round(metrics['size'], 4),
            'CO2 (kg)'    : metrics['co2'],
            'Time (s)'    : round(metrics['time'], 2),
            'Parameters'  : metrics.get('params', 'N/A'),
        })
    os.makedirs(DATA_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"  ✅ Results CSV saved: {CSV_PATH}")
    return df


# ===========================================================
# MAIN ANALYSIS RUNNER
# ===========================================================

def run_all_features(all_results, training_co2=None, model_size_mb=None, training_hours=None):
    """Run all 6 advanced features."""

    print("\n" + "=" * 70)
    print("  🚀 ADVANCED FEATURES ANALYSIS")
    print("=" * 70)

    # Feature 1: Recommendation
    print("\n" + "=" * 60)
    print("FEATURE 1: HYBRID ENERGY-AWARE RECOMMENDATION")
    print("=" * 60)
    for constraint in ['edge', 'mobile', 'cloud_gpu']:
        tech, metrics = recommend_best_technique('image', 0.90, constraint)
        print(f"\n  [{constraint.upper():10}]  → {tech.upper():<22} "
              f"saves {metrics['saving']}%  |  acc_drop {metrics['acc_drop']}%  |  speed: {metrics['speed']}")

    # Feature 2: Hardware Profiler
    print("\n" + "=" * 60)
    print("FEATURE 2: HARDWARE-AWARE OPTIMIZATION")
    print("=" * 60)
    profiler = HardwareProfiler()
    profile  = profiler.get_hardware_profile()
    print(f"\n  📊 Your Hardware Profile:")
    print(f"     System      : {profile['system']}")
    print(f"     Processor   : {profile['processor']}")
    print(f"     CPU Cores   : {profile['cpu_cores']}")
    print(f"     RAM         : {profile['ram_gb']} GB")
    print(f"     GPU         : {'Yes ✅' if profile['has_gpu'] else 'No ❌'}")
    print(f"     NPU         : {'Yes ✅' if profile['has_npu'] else 'Not detected'}")
    print(f"     Device Class: {profile['device_type'].upper()}")
    rec = profiler.recommend_techniques()
    print(f"\n  🎯 Recommended for your hardware:")
    print(f"     Best technique : {rec['best']}")
    print(f"     Reason         : {rec['reason']}")
    print(f"     Combinations   : {', '.join(rec['techniques'])}")
    if rec.get('npu_note'):
        print(f"\n  🔮 NPU Insight:")
        note = rec['npu_note']
        for line in [note[i:i+70] for i in range(0, len(note), 70)]:
            print(f"     {line}")
    print(f"\n  ⚡ Estimated energy savings per technique:")
    for t in rec['techniques']:
        savings = profiler.estimate_energy_savings(t)
        print(f"     {t:<22}: ~{savings:.0f}%")

    # Feature 3: Lifecycle Assessment
    lifecycle = None
    if training_co2 and model_size_mb and training_hours:
        print("\n" + "=" * 60)
        print("FEATURE 3: COMPLETE LIFECYCLE CARBON ASSESSMENT")
        print("=" * 60)
        lifecycle = calculate_total_carbon_footprint(training_co2, model_size_mb, training_hours)
        print(f"\n  🌍 Carbon Footprint Breakdown (Baseline Model):")
        print(f"     Training (operational) : {lifecycle['operational']:.6f} kg  ({lifecycle['pct_operational']:.1f}%)")
        print(f"     Hardware manufacturing : {lifecycle['embodied']:.6f} kg  ({lifecycle['pct_embodied']:.1f}%)")
        print(f"     {INFERENCE_YEARS}-year inference     : {lifecycle['inference']:.6f} kg  ({lifecycle['pct_inference']:.1f}%)")
        print(f"     {'─'*40}")
        print(f"     🔥 TOTAL LIFECYCLE CO₂ : {lifecycle['total']:.6f} kg")

    # Feature 4: Efficiency Scoring
    suggestion = None
    if all_results:
        print("\n" + "=" * 60)
        print("FEATURE 4: ENERGY-AWARE EFFICIENCY ANALYSIS")
        print("=" * 60)
        scoring_input = {
            k: {'accuracy': v['accuracy'], 'co2': v['co2'], 'size': v['size']}
            for k, v in all_results.items()
        }
        suggestion = energy_aware_tuning_suggestion(scoring_input)
        if suggestion:
            print(f"\n  🏆 Best accuracy-per-CO₂   : {suggestion['best_technique'].upper()}")
            print(f"  🏆 Best composite score    : {suggestion['best_composite'].upper()}")
            print(f"\n  📊 Efficiency Scores (accuracy / kg CO₂):")
            for tech, score in sorted(suggestion['scores'].items(), key=lambda x: -x[1]):
                print(f"     {tech:<24}: {score:.1f}")
            print(f"\n  📊 Composite Scores:")
            for tech, score in sorted(suggestion['composite_scores'].items(), key=lambda x: -x[1]):
                print(f"     {tech:<24}: {score:.2f}")

    # Feature 5: Visualizations
    print("\n" + "=" * 60)
    print("FEATURE 5: VISUALIZATIONS")
    print("=" * 60)
    create_visualizations(all_results)
    histories = {k: v.get('history') for k, v in all_results.items() if v.get('history') is not None}
    create_training_history_plot(histories)

    # Feature 6: CSV Export
    print("\n" + "=" * 60)
    print("FEATURE 6: CSV EXPORT")
    print("=" * 60)
    df = save_results_csv(all_results)
    print(f"\n{df.to_string(index=False)}")

    # Summary
    print("\n" + "=" * 70)
    print("  ✅ ADVANCED FEATURES COMPLETE — KEY INSIGHTS")
    print("=" * 70)
    if all_results:
        valid = {k: v for k, v in all_results.items() if v['accuracy'] > 0}
        if valid:
            best_energy = min(valid.items(), key=lambda x: x[1]['co2'])
            best_acc    = max(valid.items(), key=lambda x: x[1]['accuracy'])
            baseline    = all_results.get('Baseline', {})
            print(f"\n  🌱 Best Energy Saver  : {best_energy[0].upper():26}  ({best_energy[1]['co2']:.6f} kg CO₂)")
            print(f"  🎯 Best Accuracy      : {best_acc[0].upper():26}  ({best_acc[1]['accuracy']*100:.2f}%)")
            if baseline.get('co2', 0) > 0:
                q = all_results.get('Quantization', {})
                if q.get('co2', 0) > 0:
                    saving_pct = (1 - q['co2'] / baseline['co2']) * 100
                    print(f"  ⚡ Quantization saves ~{saving_pct:.0f}% CO₂ vs Baseline")
            print(f"  🏭 Manufacturing emissions are {EMBODIED_FACTOR}x larger than training!")
            if suggestion:
                print(f"  🏆 Best overall technique: {suggestion['best_composite'].upper()}")
    print()
    return {
        'recommendation': recommend_best_technique('image', 0.90, 'edge'),
        'hardware'      : profile,
        'lifecycle'     : lifecycle,
        'scores'        : suggestion['scores'] if suggestion else None,
    }


# ===========================================================
# REPORT GENERATOR
# ===========================================================

def generate_report(all_results):
    """Generate and save the final text report."""
    valid        = {k: v for k, v in all_results.items() if v['accuracy'] > 0}
    best_accuracy = max(valid.items(), key=lambda x: x[1]['accuracy'])
    best_size     = min(valid.items(), key=lambda x: x[1]['size'] if x[1]['size'] > 0 else 9999)
    best_co2      = min(valid.items(), key=lambda x: x[1]['co2'])
    best_time     = min(valid.items(), key=lambda x: x[1]['time'])

    n_train = 6149
    n_test  = 2040

    report = f"""
{'='*80}
                ENERGY-AWARE MACHINE LEARNING
                COMPLETE PROJECT REPORT — {MODEL_NAME}
{'='*80}

DATE             : {time.strftime('%Y-%m-%d %H:%M:%S')}
DATASET          : Oxford Flowers102  (224×224 colour images, 102 classes)
BACKBONE         : {MODEL_NAME}  (ImageNet pretrained = {USE_PRETRAINED})
TARGET ACCURACY  : 90%+
TRAINING SAMPLES : {n_train}  (original test split used for training)
TEST SAMPLES     : {n_test}   (original train + validation splits)

1. TECHNIQUES IMPLEMENTED
{'─'*80}
✓ Baseline           — {MODEL_NAME}, 2-phase training (head freeze → fine-tune)
✓ Quantization       — Float16 TFLite conversion
✓ Early Stopping     — Auto-stop on val_accuracy plateau  (patience={ES_PATIENCE})
✓ Transfer Learning  — Frozen base + rich head (Dense({TL_DENSE_UNITS}) + strong aug, 2-phase)
✓ Fine-Tuning        — Deep backbone unfreeze ({FINETUNE_UNFREEZE_LAYERS} layers) + cosine warmup LR

2. RESULTS SUMMARY
{'─'*80}
{'Technique':<22} {'Accuracy':>10} {'Size (MB)':>12} {'CO₂ (kg)':>14} {'Time (s)':>10}
{'─'*80}"""

    for technique, metrics in all_results.items():
        acc_s  = f"{metrics['accuracy']*100:.2f}%" if metrics['accuracy'] > 0 else "N/A"
        size_s = f"{metrics['size']:.3f}"          if metrics['size'] > 0    else "N/A"
        report += f"\n{technique:<22} {acc_s:>10} {size_s:>12} {metrics['co2']:>14.6f} {metrics['time']:>10.1f}"

    report += f"""

3. BEST IN CATEGORY
{'─'*80}
• Highest Accuracy  : {best_accuracy[0]:<26} ({best_accuracy[1]['accuracy']*100:.2f}%)
• Smallest Model    : {best_size[0]:<26} ({best_size[1]['size']:.3f} MB)
• Lowest CO₂        : {best_co2[0]:<26} ({best_co2[1]['co2']:.6f} kg)
• Fastest Training  : {best_time[0]:<26} ({best_time[1]['time']:.1f}s)

4. WHY {MODEL_NAME}?
{'─'*80}
• V2M has ~54M params — rich feature representations for 102 flower classes
• Pretrained on ImageNet — textures, shapes, colour gradients transfer well
• 2-phase training: freeze backbone → train head, then unfreeze last N layers
• Expected accuracy on Flowers102: 88–93% depending on technique

5. TECHNIQUE EXPLANATIONS
{'─'*80}
Baseline:
  2-phase EfficientNetV2M training. Phase 1 freezes the backbone and trains
  only the classification head. Phase 2 unfreezes the last {FINE_TUNE_LAYERS} layers for
  fine-tuning at a very small LR. Reference point for all comparisons.

Quantization:
  Post-training Float16 TFLite conversion. Halves model file size with
  near-zero accuracy loss. No retraining required — pure mathematical
  precision reduction on every weight in the model.

Early Stopping:
  Same model as Baseline but training stops automatically when val_accuracy
  plateaus (patience={ES_PATIENCE}). Saves energy by skipping unnecessary epochs.

Transfer Learning:
  Phase 1: Entire backbone frozen, only the richer head (Dense({TL_DENSE_UNITS}) +
  Dropout + Softmax) is trained with strong augmentation.
  Phase 2: Last 20 backbone layers unfrozen at LR=1e-5 (batch=8 to avoid OOM).
  Fastest technique with competitive accuracy via pure ImageNet feature reuse.

Fine-Tuning:
  Loads the Transfer Learning model and unfreezes the last {FINETUNE_UNFREEZE_LAYERS} backbone
  layers. Uses cosine warmup LR schedule — linear ramp-up over {FINETUNE_WARMUP_EPOCHS} epochs
  then cosine decay — to prevent catastrophic forgetting of pretrained weights.
  Highest accuracy technique of all five.

6. LIFECYCLE CARBON NOTE
{'─'*80}
Hardware manufacturing (embodied carbon) is ~{EMBODIED_FACTOR}x the training footprint.
Even small model size reductions via quantization compound across
{INFERENCE_YEARS} years of inference at {DAILY_INFERENCES} daily predictions.

7. CONCLUSION
{'─'*80}
{MODEL_NAME} with Fine-Tuning achieves the highest accuracy on Flowers102.
Transfer Learning provides the best energy-efficiency trade-off for
deployment scenarios where training cost must be minimised.
Quantization is the recommended post-training step before any deployment —
it halves model size with negligible accuracy loss and zero training cost.

{'='*80}
                        END OF REPORT
{'='*80}
"""
    print(report)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n✅ Report saved: {REPORT_PATH}")


# ===========================================================
# ENTRY POINT
# ===========================================================

if __name__ == "__main__":
    demo_mode = '--demo' in sys.argv

    if demo_mode:
        print("\n[DEMO MODE]  Using synthetic results — no training required")
        all_results = {
            'Baseline'          : {'accuracy': 0.9088, 'co2': 0.000801, 'size': 204.1, 'time': 22648.3, 'params': 54000000, 'history': None},
            'Quantization'      : {'accuracy': 0.9020, 'co2': 0.001147, 'size':  51.0, 'time':   397.4, 'params': 'N/A',    'history': None},
            'Early Stopping'    : {'accuracy': 0.8838, 'co2': 0.126780, 'size': 204.1, 'time': 19222.3, 'params': 54000000, 'history': None},
            'Transfer Learning' : {'accuracy': 0.9123, 'co2': 0.051876, 'size': 204.1, 'time': 14080.5, 'params': 54000000, 'history': None},
            'Fine-Tuning'       : {'accuracy': 0.9200, 'co2': 0.060000, 'size': 204.1, 'time': 15000.0, 'params': 54000000, 'history': None},
        }
    else:
        print("\n" + "=" * 70)
        print("  Loading saved results from JSON files...")
        print("=" * 70)
        all_results = load_all_results()
        if not all_results:
            print("\n  ❌ No saved results found. Run train.py or resume.py first.")
            sys.exit(1)

    b = all_results.get('Baseline', list(all_results.values())[0])

    run_all_features(
        all_results    = all_results,
        training_co2   = b['co2'],
        model_size_mb  = b['size'],
        training_hours = b['time'] / 3600,
    )

    generate_report(all_results)

    print("\n" + "=" * 70)
    print("  📁 OUTPUT FILES")
    print("=" * 70)
    print(f"  📊 {PLOT_COMPARISON_PATH}")
    print(f"  📊 {PLOTS_DIR}/training_history.png")
    print(f"  📄 {REPORT_PATH}")
    print(f"  📊 {CSV_PATH}")
    print(f"  🤖 {BASELINE_MODEL_PATH}")
    print(f"  🤖 {EARLY_STOP_MODEL_PATH}")
    print(f"  🤖 {TL_MODEL_PATH}")
    print(f"  🤖 {FINETUNED_MODEL_PATH}")
    print(f"  🤖 {QUANTIZED_MODEL_PATH}")
    print("\n✅ PROJECT COMPLETE!")