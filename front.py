import os

from ultralytics import YOLO


def find_newest_model() -> YOLO:
    # get all dirs of runs/train
    return YOLO("model.pt")


def predict(urls, model: YOLO, show_images: bool) -> list[int]:

    for url in urls:
        results = model.predict(
            url,
            conf=0.1,
            verbose=False,
            show_labels=False,
            show_conf=False,
            line_width=1,
            augment=False,
        )

    counts = []

    # Visualize the results
    for r in results:
        counts.append(len(r.boxes))
        if show_images:
            r.show()

    return counts


if __name__ == "__main__":
    urls = [
        "https://images.pexels.com/photos/3289479/pexels-photo-3289479.jpeg?cs=srgb&dl=high-angle-shot-of-cars-and-people-on-the-road-3289479.jpg"
    ]

    SHOW_IMAGES = True

    for i, url in enumerate(urls):
        res_base = predict([url], YOLO(verbose=False), show_images=SHOW_IMAGES)
        res_new = predict([url], find_newest_model(), show_images=SHOW_IMAGES)
        even_newer = predict([url], YOLO("best.pt"), show_images=SHOW_IMAGES)
        print(f"Predicting image {i}")
        print(f"Base model: {res_base[0]}")
        print(f"Newest model: {res_new[0]}")
        print(f"Even newer model: {even_newer[0]}")
        print("\n")
