import marimo

__generated_with = "0.1.77"
app = marimo.App()


@app.cell
def __(mo):
    mo.md("# Is it a bird? Creating a model from your own data")
    return


@app.cell
def __():
    from duckduckgo_search import DDGS
    from fastcore.all import L

    def search_images(term, max_images=30):
        print(f"Searching for '{term}'")
        return L(DDGS().images(term, max_results=max_images)).itemgot("image")
    return DDGS, L, search_images


@app.cell
def __(mo, search_images):
    urls = search_images("bird photos", max_images=1)
    mo.md(urls[0])
    return urls,


@app.cell
def __(mo):
    mo.md("...and then download a URL and take a look at it:")
    return


@app.cell
def __(download_url, mo, urls):
    _dest = "bird.jpg"
    download_url(urls[0], _dest, show_progress=False)

    # from fastai.vision.all import Image

    # im = Image.open(dest)
    mo.image(src=_dest, width=256, height=256)
    return


@app.cell
def __(mo):
    mo.md("Now let's do the same with 'forest photos'")
    return


@app.cell
def __(download_url, mo, search_images):
    download_url(
        search_images("forest photos", max_images=1)[0],
        "forest.jpg",
        show_progress=False,
    )
    mo.image(src="forest.jpg", width=256, height=256)
    return


@app.cell
def __(mo):
    mo.md(
        'Our searches seem to be giving reasonable results, so let\'s grab a few examples of each of "bird" and "forest" photos, and save each group of photos to a different folder (I\'m also trying to grab a range of lighting conditions here):'
    )
    return


@app.cell
def __(Path, download_images, resize_images, search_images):
    searches = "forest", "bird"
    output = Path("bird_or_not")

    from time import sleep

    for topic in searches:
        dest = output / topic
        dest.mkdir(exist_ok=True, parents=True)
        download_images(dest, urls=search_images(f"{topic} photo"))
        sleep(10)
        download_images(dest, urls=search_images(f"{topic} sun photo"))
        sleep(10)
        download_images(dest, urls=search_images(f"{topic} shade photo"))
        resize_images(dest, max_size=400, dest=dest)
    print("done")
    datadir = output
    return datadir, dest, output, searches, sleep, topic


@app.cell
def __(Path, datadir, get_image_files, verify_images):
    failed = verify_images(get_image_files(datadir))
    failed.map(Path.unlink)
    print(len(failed))
    return failed,


@app.cell
def __(
    data_block,
    data_transforms,
    datadir,
    vision_augment,
    vision_data,
    vision_utils,
):
    dls = data_block.DataBlock(
        blocks=(vision_data.ImageBlock, data_block.CategoryBlock),
        get_items=vision_utils.get_image_files,
        splitter=data_transforms.RandomSplitter(valid_pct=0.2, seed=42),
        get_y=data_transforms.parent_label,
        item_tfms=[vision_augment.Resize(192, method="squish")],
    ).dataloaders(datadir, bs=32)

    dls.show_batch(max_n=6)
    return dls,


@app.cell
def __(dls, vision_all):
    learn = vision_all.vision_learner(dls, vision_all.resnet18, metrics=vision_all.error_rate)
    learn.fine_tune(3)
    return learn,


@app.cell
def __(learn, vision_all):
    _is_bird,_,_probs = learn.predict(vision_all.PILImage.create("bird.jpg"))
    print(f"This is a: {_is_bird}")
    print(f"Probability it's a bird: {_probs[0]:.4f}")
    return


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __():
    from pathlib import Path
    return Path,


@app.cell
def __():
    from fastdownload import download_url
    from fastai.vision.utils import (
        download_images,
        resize_images,
        verify_images,
        get_image_files,
    )
    from fastai.vision import (
        augment as vision_augment,
        data as vision_data,
        utils as vision_utils,
    )
    from fastai.vision import all as vision_all
    from fastai.data import block as data_block, transforms as data_transforms
    return (
        data_block,
        data_transforms,
        download_images,
        download_url,
        get_image_files,
        resize_images,
        verify_images,
        vision_all,
        vision_augment,
        vision_data,
        vision_utils,
    )


if __name__ == "__main__":
    app.run()
