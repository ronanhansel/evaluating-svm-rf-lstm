import matplotlib.pyplot as plt
import numpy as np

# Data preparation
models = ['RF', 'LSTM', 'SVM']
metrics = ['Accuracy', 'F1', 'Precision', 'Recall']

data = {
    'RF': [0.9658, 0.9696, 0.9696, 0.9696],
    'LSTM': [0.9645, 0.9645, 0.9647, 0.9645],
    'SVM': [0.9093, 0.9092, 0.9096, 0.9093]
}

x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 8))
rects1 = ax.bar(x - width, data['RF'], width, label='RF', color='#2ecc71')
rects2 = ax.bar(x, data['LSTM'], width, label='LSTM', color='#3498db')
rects3 = ax.bar(x + width, data['SVM'], width, label='SVM', color='#e74c3c')

# Set logarithmic scale
ax.set_yscale('log')
ax.set_ylim([0.9, 1.0])  # Adjust limits to focus on range of values

# Customization
ax.set_ylabel('Score (log scale)')
ax.set_title('Model Performance Comparison (Logarithmic Scale)')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(True, alpha=0.3)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=90)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.subplots_adjust(top=0.9)
plt.tight_layout()
plt.savefig('plots/model_comparison_log.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.close()