



# After running inference

# Load the model
src_sample, tgt_sample = get_sample_from_dataloader(val_dataloader, device)
with torch.no_grad():
    max_seq_length = src_sample.size(1)
    predictions, attn_probs = model(src_sample, tgt_sample, src_mask, tgt_mask)

# Feature names (example)
feature_names = ['Temperature', 'Humidity', 'WindSpeed', 'SolarRadiation', 'HeatEnergy']

# PLOT HEATMAP
plot_attention_heatmap(attn_probs, src_sample, feature_names)
