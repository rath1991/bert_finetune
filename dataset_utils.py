"""Include dataset class and methods for data-loader"""
import torch
from torch.utils.data import DataLoader, Dataset


class ReviewDataset(Dataset):
    """Class that converts reviews to torch appropriate dataset.

    Args:
      reviews: Product reviews.
      targets: Actual sentiment scores.
      tokenizer: Bert tokenizer.
      max_len: Maximum allowable length for encoding.

    Returns:
      Dictionary of reviews, input_ids, attention_mask, targets.

    """

    def __init__(self, reviews: str, targets: int, tokenizer, max_len: int):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        targets = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation = True, 
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "review_text": review,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": torch.tensor(targets, dtype=torch.long),
        }


def create_data_loader(data, tokenizer, max_len, batch_size):
    """Function that creates dataloader for training and eval.

    Args:
      data: Review data.
      tokenizer: Bert tokenizer.
      max_len: Maximum allowable length for encoding.
      batch_size: Batch size of data for training and evaluation.
    """
    dataset = ReviewDataset(
        reviews=data.content.to_numpy(),
        targets=data.sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
    )

    return DataLoader(dataset, batch_size=batch_size, num_workers=4)

def to_sentiment(rating: int) -> int:
  """Convert review rating to sentiment.
  
  Args:
    rating: Ratings for a product.
  
  Returns:
    sentiment: Converted sentiment for the product.
  """
  if rating <=2:
    sentiment = 0
  elif rating == 3:
    sentiment = 1
  else:
    sentiment = 2
  return sentiment
