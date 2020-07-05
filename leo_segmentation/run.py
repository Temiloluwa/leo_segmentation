# Entry point for the project
import torch

def train_model(saved_model_path:str):
    print(f"I am training a model saved at {saved_model_path}")
    return

def main():
    model_path = "model_path"
    train_model(model_path)

if __name__ == "__main__":
    main()