"""Conducts finetuning of bert llm."""
import pandas as pd 
import numpy as np
import torch
from torch import nn, optim
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from dataset_utils import ReviewDataset
from dataset_utils import (create_data_loader, to_sentiment)
from model_utils import SentimentClassifier
from model_utils import (train_epoch, eval_model, run_model)
import os 
import warnings 
warnings.filterwarnings('ignore')
from transformers import logging
logging.set_verbosity_error()

if __name__ == '__main__':
    #Hardcoded parameters to be placed under config.json file later.
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device,{device}')
    #----------------------------------------#
    train_val_split = 0.1
    test_val_split = 0.5
    #----------------------------------------#
    BATCH_SIZE = 16
    MAX_LEN = 160
    EPOCHS = 1
    LR = 2e-5
    MODEL_NAME = 'bert-base-cased'
    #-------------------------------------------#

    df = pd.read_csv('reviews.csv')
    #create new sentiment column
    df['sentiment'] = df['score'].apply(lambda x: to_sentiment(x))
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')
    path = os.getcwd()
    #-------------------------------------------#
    #training validation and test data creation
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)

    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

    bert_model = BertModel.from_pretrained(MODEL_NAME)

    model = SentimentClassifier(
        len(df['sentiment'].unique()),
        bert_model
        )
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr= LR, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(device)
    print('starting finetuning')
    run_model(EPOCHS,
        model,
        train_data_loader,
        val_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(df_train),
        len(df_val),
        path)

