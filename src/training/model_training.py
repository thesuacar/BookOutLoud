from src.preprocessing.preprocessing_utils import FlickrDataset, clean_captions_txt, build_vocab, transforms
from src.training.training_utils import train_model, evaluate_bleu_fn
from src.training.encoders import EncoderCNN,DecoderRNN
import torch
import nltk
nltk.download('punkt', quiet=True)

def main():
    # Paths to data
    image_path = "data/images/"
    caption_path = "data/captions.txt"

    # Clean text and prepare dataset
    df = clean_captions_txt(image_path, caption_path)

    # Tokenize captions
    df['tokens'] = df['caption'].apply(nltk.word_tokenize)

    # Build vocabulary
    vocab = build_vocab(df['tokens'].tolist(), vocab_size=5000)

    # Encode captions
    df['encoded'] = df['tokens'].apply(lambda tokens: [vocab.get(token.lower(), vocab["<UNK>"]) for token in tokens])
    
    # Define transformations for images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Create dataset and dataloaders
    dataset = FlickrDataset(df, image_path, transform=transform)
    train_size = int(0.8 * len(dataset))
    dev_size = len(dataset) - train_size
    train_dataset, dev_dataset = torch.utils.data.random_split(dataset, [train_size, dev_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=32)

    # Initialize models, criterion, optimizer
    encoder = EncoderCNN(embed_size=256)
    decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=len(vocab))
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

    # Train the model
    train_model(
        encoder,
        decoder,
        train_loader,
        dev_loader,
        criterion,
        optimizer,
        num_epochs=10,
        vocab_size=len(vocab),
        evaluate_bleu_fn=evaluate_bleu_fn,
        eval_interval=1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )