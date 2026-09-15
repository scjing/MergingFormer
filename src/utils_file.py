import matplotlib.pyplot as plt
import torch

def plot_loss_curve(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue', marker='o')
    plt.plot(val_losses, label='Validation Loss', color='orange', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def save_model_weights(model, save_path):
    """
    保存模型的权重

    Parameters:
        model (torch.nn.Module): 要保存权重的模型
        save_path (str): 保存权重的路径
    """
    torch.save(model.state_dict(), save_path)
