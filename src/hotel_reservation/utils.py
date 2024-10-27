import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_results(cm):
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted Reservation")
    plt.ylabel("Actual Reservation")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_importance, feature_names, top_n=10):
    plt.figure(figsize=(10, 6))
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx[-top_n:].shape[0]) + .5
    plt.barh(pos, feature_importance[sorted_idx[-top_n:]])
    plt.yticks(pos, feature_names[sorted_idx[-top_n:]])
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    plt.show()