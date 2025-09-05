import matplotlib.pyplot as plt
def visualize_attention(model, img):
    model.eval()
    with torch.no_grad():
        outputs = model(img.unsqueeze(0))  # Unsqueeze for batch dimension
        attentions = model.vit.encoder.layer[-1].attention.attention_probs  # Last layer attention
    attn = attentions.mean(1).squeeze(0).cpu().detach().numpy()
    plt.imshow(attn[0], cmap='hot')
    plt.title("Attention Map")
    plt.show()
