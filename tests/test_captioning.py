from app.image_captioning import load_captions, prepare_subset, get_image_transform, CaptionDataset

images_path = "data/images"
captions_file = "data/captions.txt"

captions = load_captions(images_path, captions_file)
subset = prepare_subset(captions)
transform = get_image_transform()

dataset = CaptionDataset(subset, images_path, transform)
print(len(dataset))
