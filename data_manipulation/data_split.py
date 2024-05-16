import os

# in /datasets/data/sec1 are files .png .txt, split to labels and images folders


# path = "./datasets/data/sec1/"
# files = os.listdir(path)
# files = [f for f in files if f.endswith(".png")]

# for fille in files:
#     name = fille.split(".")[0]
#     os.rename(path + fille, f"{path}images/{name}.png")
#     os.rename(path + name + ".txt", f"{path}labels/{name}.txt")

# print("Done")

# now do 80/20 split
import random
import shutil

path = "./datasets/data/sec1/"
# files = os.listdir(f"{path}images/")
# random.shuffle(files)

# split = int(0.8 * len(files))
# train_files = files[:split]
# test_files = files[split:]

# for file in train_files:
#     shutil.move(f"{path}images/{file}", f"{path}train/images/{file}")
#     shutil.move(
#         f"{path}labels/" + file.replace(".png", ".txt"),
#         f"{path}train/labels/" + file.replace(".png", ".txt"),
#     )

# for file in test_files:
#     shutil.move(f"{path}images/{file}", f"{path}test/images/{file}")
#     shutil.move(
#         f"{path}labels/" + file.replace(".png", ".txt"),
#         f"{path}/test/labels/" + file.replace(".png", ".txt"),
#     )


for label_file in os.listdir(f"{path}valid/labels/"):
    with open(f"{path}valid/labels/{label_file}", "r") as f:
        lines = f.readlines()
        new_lines = []
        for line in lines:
            if len(line) < 5:
                continue
            new_lines.append(f"0{line[1:]}")

    with open(f"{path}valid/labels/{label_file}", "w") as f:
        f.writelines(new_lines)


for label_file in os.listdir(f"{path}train/labels/"):
    with open(f"{path}train/labels/{label_file}", "r") as f:
        lines = f.readlines()
        new_lines = []
        for line in lines:
            if len(line) < 5:
                continue
            new_lines.append(f"0{line[1:]}")

    with open(f"{path}train/labels/{label_file}", "w") as f:
        f.writelines(new_lines)
