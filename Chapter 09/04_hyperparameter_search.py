from transformers import TrainingArguments, Trainer
import optuna

# Define search space
def hp_space(trial):
    return {
        "learning_rate": trial.suggest_loguniform("learning_rate", 2e-5, 5e-5),
        "per_device_train_batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "weight_decay": trial.suggest_uniform("weight_decay", 0.0001, 0.01),
    }

from transformers import AutoModelForSequenceClassification

# Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=100,
)

# Run hyperparameter search
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

best_trial = trainer.hyperparameter_search(
    direction="maximize", 
    hp_space=hp_space
)
print("Best Hyperparameters:", best_trial)
