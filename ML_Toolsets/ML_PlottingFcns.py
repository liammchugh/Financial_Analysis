import matplotlib.pyplot as plt
import seaborn as sns


def plot_attention_heatmap(attn_weights, input_seq, feature_names):
    """
    Generates heatmaps based on attention weights.

    Parameters:
    - attn_weights: The attention weights returned by the transformer model.
    - input_seq: The input time series data.
    - feature_names: List of feature names to label the heatmap axes.
    """
    # Assuming attn_weights shape is (batch_size, num_heads, seq_len, seq_len)
    attn_weights = attn_weights.mean(dim=1)  # Average across all heads (shape: batch_size, seq_len, seq_len)
    attn_weights = attn_weights.squeeze(0).detach().cpu().numpy()  # Remove batch dimension
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(attn_weights, xticklabels=feature_names, yticklabels=range(len(input_seq)), cmap='coolwarm', ax=ax)
    ax.set_xlabel('Feature')
    ax.set_ylabel('Time Step')
    ax.set_title('Attention Heatmap')
    plt.show()

