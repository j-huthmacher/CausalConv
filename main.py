"""
    @autor: j-huthmacher
"""
import argparse
from pathlib import Path
import yaml
from datetime import datetime
import logging

import torch
from torch import nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from load_data import load_data, split_data
from model import CausalConvNet, MLP
from visualization.plots import eval_plot, create_gif


#### CLI ####
parser = argparse.ArgumentParser(prog='CausalConv', description='Causal Convolutional Neural Net (TCN)')

parser.add_argument('--config', dest='config',
                    help='Defines which configuration should be used from config.yml.')

parser.add_argument('--path', dest='path',
                    help='Output path.')


def evaluate(predictions, labels):
    """ Evaluation function.

        Currently only accuracy is implemented
    """
    k = 1  # Top-k
    correct = np.sum([l in pred for l, pred in zip(labels, np.asarray(predictions)[:,:k])])
    top1 = (correct/len(labels))

    return top1

#### Main Program ####
args = parser.parse_args()


#### Load Config ####
config = "train" if args.config is None else args.config  # Default config is train
with open("config.yml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)[config]

date = datetime.now().strftime("%d%m%Y-%H%M")
output = f"./output/{date}/" if args.path is None else args.path
Path(output).mkdir(parents=True, exist_ok=True)

#### Logger ####
logging.basicConfig(filename=f"{output}log.txt", level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s', datefmt='%d.%m.%Y %H:%M:%S')

#### Load Data ####
x, y = load_data(**config["data"])
x = x.to_numpy().reshape(-1, config["causal_conv"]["c_in"], config["causal_conv"]["seq_length"])
y = y[::config["causal_conv"]["seq_length"]]  # Create lagged outputs

train, val = split_data(x, y.to_numpy())
train_loader = DataLoader(train, **config["loader"])
val_loader = DataLoader(val, **config["loader"])


#### Create Model #####
model = CausalConvNet(**config["causal_conv"]).cuda()
classifier = MLP(num_class=np.max(y) + 1,**config["mlp"]).cuda()

#### Tracking Varaibales #####
train_batch_losses = []
train_loss = []
train_metrics = []
train_pred = np.array([])
train_label = np.array([])

val_batch_losses = []
val_loss = []
val_metrics = []
val_pred = np.array([])
val_label = np.array([])


#### Training ####
loss_fn = nn.CrossEntropyLoss()
optimizer = getattr(optim, config["optimizer"])(list(model.parameters()) + list(classifier.parameters()),
                                                **config["optimizer_cfg"])

with open(f'{output}/config.yml', 'w') as outfile:
    yaml.dump(config, outfile, default_flow_style=False)

try:
    for epoch in trange(config["n_epochs"], desc="Epochs"):
        #### Training ####
        model.train()
        for batch_x, batch_y in train_loader:
            emb = model(batch_x)
            yhat = classifier(emb)
            loss = loss_fn(yhat, batch_y.type("torch.LongTensor").to(model.device))

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            #### EVALUATION DURING TRAINING ####
            yhat_idx = torch.argsort(yhat, descending=True)
            train_pred = np.vstack([train_pred, yhat_idx.detach().cpu().numpy()]) if train_pred.size else yhat_idx.detach().cpu().numpy()
            train_label = np.append(train_label, batch_y.detach().cpu().numpy())

            train_batch_losses.append(torch.squeeze(loss).item())

        train_metric = evaluate(train_pred, train_label)
        train_metrics.append(train_metric)

        train_loss.append(np.mean(train_batch_losses) if len(train_batch_losses) > 0 else 0)
        train_batch_losses = []
        
        #### Validation #####
        with torch.no_grad():
            model.eval()
            for batch_x, batch_y in train_loader:
                yhat = model(batch_x)
                loss = loss_fn(yhat, batch_y.type("torch.LongTensor").to(model.device))

                #### EVALUATION DURING TRAINING ####
                yhat_idx = torch.argsort(yhat, descending=True)
                val_pred = np.vstack([val_pred, yhat_idx.detach().cpu().numpy()]) if val_pred.size else yhat_idx.detach().cpu().numpy()
                val_label = np.append(val_label, batch_y.detach().cpu().numpy())

                val_batch_losses.append(torch.squeeze(loss).item())

            val_metric = evaluate(val_pred, val_label)
            val_metrics.append(val_metric)

        
        val_loss.append(np.mean(val_batch_losses) if len(val_batch_losses) > 0 else 0)
        val_batch_losses = []

        #### Create plots ####
        if "plot" in config and config["plot"]:
            fig = eval_plot(x, y, model, classifier, 
                            {"train loss": train_loss, "val loss": val_loss},
                            {"train acc": train_metrics, "val acc": val_metrics},
                            n_epochs=config["n_epochs"], prec=5e-4,model_name="CausalConv - MLP Classifier")
            plt.close()
            create_gif(fig, path=output+"CausalConv.Train.Val.gif", fill=config["n_epochs"] -1 != epoch)


    #### Store Results ####
    np.save(f'{output}/CuasalConv_train_losses.npy', train_loss)
    np.save(f'{output}/CuasalConv_train_metrics.npy', train_metrics)
    with open(f'{output}/config.yml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

except Exception as e:
    print(e)
    logging.exception(e)
    
