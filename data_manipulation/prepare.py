import random
import os

os.makedirs("valid/images", exist_ok=True)
os.makedirs("valid/labels", exist_ok=True)
os.makedirs("train/images", exist_ok=True)
os.makedirs("train/labels", exist_ok=True)

files = []

for folder in os.listdir("./"):
    if folder in ["valid", "train", "prepare.py"]:
        continue

    for file in os.listdir(folder):

        if file.endswith(".png"):
            files.append(f"{folder}/{file}")


random.shuffle(files)

for i, file in enumerate(files):

    filename = file.split("/")[-1]

    if i < len(files) * 0.2:
        os.rename(file, f"valid/images/{filename}")
        os.rename(
            file.replace(".png", ".txt"),
            f"valid/labels/{filename.replace('.png', '.txt')}",
        )
    else:
        os.rename(file, f"train/images/{filename}")
        os.rename(
            file.replace(".png", ".txt"),
            f"train/labels/{filename.replace('.png', '.txt')}",
        )

# make every label line start with "0"


for folder in ["valid/labels", "train/labels"]:
    for file in os.listdir(folder):
        with open(f"{folder}/{file}", "r") as f:
            lines = f.readlines()

        with open(f"{folder}/{file}", "w") as f:
            for line in lines:
                f.write(f"0{line[1:]}")
