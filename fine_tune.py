import json
import random
from typing import List, Dict
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
import os

class ComplexityPattern:
    def __init__(self, code: str, complexity: str, description: str):
        self.code = code
        self.complexity = complexity
        self.description = description

class CodeDatasetGenerator:
    def __init__(self):
        self.patterns = {
            "O(1)": [
                ComplexityPattern(
                    """def get_first(arr):
    if len(arr) > 0:
        return arr[0]
    return None""",
                    "O(1)",
                    "Constant time array access"
                ),
                ComplexityPattern(
                    """def is_empty(stack):
    return len(stack) == 0""",
                    "O(1)",
                    "Simple length check"
                )
            ],
            "O(n)": [
                ComplexityPattern(
                    """def find_max(numbers):
    if not numbers:
        return None
    max_num = numbers[0]
    for num in numbers:
        if num > max_num:
            max_num = num
    return max_num""",
                    "O(n)",
                    "Linear search for maximum"
                ),
                ComplexityPattern(
                    """def count_elements(arr):
    count = 0
    for _ in arr:
        count += 1
    return count""",
                    "O(n)",
                    "Linear counting"
                )
            ],
            "O(n²)": [
                ComplexityPattern(
                    """def has_duplicates(arr):
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] == arr[j]:
                return True
    return False""",
                    "O(n²)",
                    "Nested loop duplicate check"
                ),
                ComplexityPattern(
                    """def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr""",
                    "O(n²)",
                    "Selection sort"
                )
            ]
        }

    def generate_sample(self) -> Dict[str, str]:
        complexity = random.choice(list(self.patterns.keys()))
        pattern = random.choice(self.patterns[complexity])
        return {
            "code": pattern.code,
            "complexity": pattern.complexity,
            "description": pattern.description
        }

    def generate_dataset(self, size: int) -> List[Dict[str, str]]:
        return [self.generate_sample() for _ in range(size)]

    def augment_dataset(self, samples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        augmented_samples = []
        for sample in samples:
            augmented_samples.append(sample)
            # Add version with description as comment
            commented_code = f"# {sample['description']}\n{sample['code']}"
            augmented_samples.append({
                **sample,
                "code": commented_code
            })
        return augmented_samples

class CodeDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.complexity_map = {
            "O(1)": 0,
            "O(n)": 1,
            "O(n²)": 2
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["code"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.complexity_map[item["complexity"]])
        }

def train_complexity_model(
    output_dir: str = "./model",
    dataset_size: int = 20000,
    num_epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 1e-5
):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate and prepare dataset
    generator = CodeDatasetGenerator()
    initial_data = generator.generate_dataset(dataset_size)
    augmented_data = generator.augment_dataset(initial_data)
    train_data, eval_data = train_test_split(augmented_data, test_size=0.2, random_state=42)

    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = RobertaForSequenceClassification.from_pretrained(
        "microsoft/codebert-base",
        num_labels=3,
        problem_type="single_label_classification"
    )

    # Create datasets
    train_dataset = CodeDataset(train_data, tokenizer)
    eval_dataset = CodeDataset(eval_data, tokenizer)

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        learning_rate=learning_rate,
        gradient_accumulation_steps=2
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    # Train the model
    trainer.train()

    # Save the trained model and tokenizer
    model.save_pretrained(os.path.join(output_dir, "complexity_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "complexity_tokenizer"))

    # Save complexity mapping
    reverse_complexity_map = {str(v): k for k, v in train_dataset.complexity_map.items()}
    with open(os.path.join(output_dir, "complexity_mapping.json"), "w") as f:
        json.dump(reverse_complexity_map, f, indent=2)

    print("Training completed and model saved!")

if __name__ == "__main__":
    train_complexity_model()