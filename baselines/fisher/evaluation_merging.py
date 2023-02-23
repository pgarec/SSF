import torch
import hydra
from model_merging.data import MNIST, load_models
from merge_fisher import evaluate_fisher
from merge_isotropic import evaluate_isotropic
import torch.nn.functional as F
import matplotlib.pyplot as plt


def plot_results(results):
    plt.bar(list(results.keys()), list(results.values()))
    plt.xlabel("Type of merging")
    plt.ylabel("Average Test Loss")
    plt.xticks(list(results.keys()))
    plt.show()


@hydra.main(config_path="./configurations", config_name="merge.yaml")
def evaluate_techniques(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(cfg)
    dataset = MNIST(cfg)
    cfg.data.batch_size_test = 1
    test_loader = dataset.create_inference_dataloader()
    y_classes = dict(zip(cfg.data.digits, range(len(cfg.data.digits))))
    criterion = torch.nn.CrossEntropyLoss()
    val_loss = 0
    outputs = []

    isotropic_loss, _ = evaluate_isotropic(cfg, models, test_loader, criterion)
    fisher_loss, _ = evaluate_fisher(cfg, models, test_loader, criterion)

    with torch.no_grad():
        for _, (x, y) in enumerate(test_loader):
            batch_onehot = y.apply_(lambda i: y_classes[i])
            output_model = []
            for m in models:
                m.to(device)
                m.eval()
                out = m(x.to(device))
                output_model.append(out)
            
            outputs = torch.stack(output_model, dim=0).sum(dim=0)
            outputs = outputs / len(models)
            loss = criterion(outputs, F.one_hot(batch_onehot, cfg.data.n_classes).to(torch.float))
            val_loss += loss

    output_loss = val_loss / len(test_loader)

    results = {'isotropic loss': sum(isotropic_loss)/len(test_loader), 'fisher_loss': sum(fisher_loss)/len(test_loader), 'output_loss': output_loss}
    print(results)
    plot_results(results)


if __name__ == "__main__":
    evaluate_techniques()
