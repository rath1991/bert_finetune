"""Utils file containing all model class and functions."""
import numpy as np
import torch
from torch import nn, optim
from collections import defaultdict
from datetime import datetime

class SentimentClassifier(nn.Module):
    """This class builds sentiment classifier model over pretrained bert.

    Args:
      n_classes: Number of sentiment classes.
      model: name of Bert model being used.
    """

    def __init__(self, n_classes, model):
        """Class initializer"""
        super(SentimentClassifier, self).__init__()
        self.bert = model
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        """Conduct forward pass."""
        pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        output = self.drop(pooled_output.pooler_output)
        return self.out(output)


def train_epoch(
    model, data_loader, loss_fn, optimizer, device, scheduler, n_examples
) -> None:
    """Training function that trains the model.

    Args:
      model: Sentiment classifer model built on bert.
      data_loader: Training dataloader from torch.
      loss_fn: Cross entropy loss function.
      Optimizer: Type of optimizer used.
      device: Ensures transformation of data if GPU used.
      scheduler: Adapts and varies learning rate.
      n_examples: Total number of samples to train on.
    """

    model = model.train()

    losses = []
    correct_predictions = 0

    for data in data_loader:
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        targets = data["targets"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        # to prevent accumulation of gradient for next iteration
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    """Training function that trains the model.

    Args:
      model: Sentiment classifer model built on bert.
      data_loader: Training dataloader from torch.
      loss_fn: Cross entropy loss function.
      device: Ensures transformation of data if GPU used.
      n_examples: Total number of samples to train on.
    """
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for data in data_loader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            targets = data["targets"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


def run_model(
    epochs,
    model,
    train_data_loader,
    val_data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    train_examples,
    val_examples,
    path
) -> None:
    """Launches training based on provided config.

    Args:
      epochs: Number of epochs to finetune model on.
      model: Sentiment classifer model built on bert.
      train_data_loader: Training dataloader from torch.
      val_data_loader : Validation dataloader from torch.
      loss_fn: Cross entropy loss function.
      optimizer: Type of optimizer
      device: Ensures transformation of data if GPU used.
      scheulder: Adjust learning rate.
      train_examples: Total number of examples to train on.
      val_examples : Total number of validation examples.
      path: Specified path to save model.
    """
    start = datetime.now()
    history = defaultdict(list)
    best_accuracy = 0

    out_f = open('logfile.txt', 'w', encoding = "utf-8")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}\n")
        out_f.write(f"Epoch {epoch + 1}/{epochs}\n")
        print("-" * 10)
        out_f.write(f"---------------\n" )
        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            train_examples,
        )

        print(f"Train loss {train_loss} accuracy {train_acc}\n")
        out_f.write(f"Train loss {train_loss} accuracy {train_acc}\n")
        val_acc, val_loss = eval_model(
            model, val_data_loader, loss_fn, device, val_examples
        )

        print(f"Valloss {val_loss} accuracy {val_acc}\n")
        out_f.write(f"Valloss {val_loss} accuracy {val_acc}\n")

        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), path)
            best_accuracy = val_acc
    end = datetime.now()
    print(f'Total time for finetuning,{end-start} seconds')
    out_f.write(f'Total time for finetuning,{end-start} seconds\n')
    out_f.close()
