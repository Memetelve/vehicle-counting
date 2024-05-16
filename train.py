from ultralytics import YOLO
import torch


def main():
    # model_config = "data/data.yaml"
    model_config = "data/data_big.yaml"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} for training")

    # Initialize model
    model = YOLO()

    # Train model
    model.train(
        data=model_config,
        epochs=20,
        batch=24,
        imgsz=640,
        name="exp",
        optimizer="auto",
        lr0=0.0001,
        lrf=0.1,
        dropout=0.1,
        workers=0,
    )


if __name__ == "__main__":
    main()
