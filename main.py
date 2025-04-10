import argparse
import split_data
import train_ViT_classification
import build_fine_cams
import train_CNN_decoder
import build_masks
import train_supervised_gt
import train_supervised_ws
import test_results
from utils.utils import set_seed
from torchvision import transforms

def main(train: bool):
    seed = 42
    set_seed(seed)
    vit_model = 'tiny'
    decoder_size = "medium"
    gt_ratio = 0.0
    # loss_threshold = 0.05  best for small medium use_finecam_only=True loss_threshold=0.01 10 epochs training_supervised_ws
    # loss_threshold = 0.35  best for tiny medium use_finecam_only=False 4 epochs training_supervised_ws
    # loss_threshold = 0.25 best for tiny large use_finecam_only=False 10 epochs training_supervised_ws
    # loss_threshold = 0.01  # best for small medium use_finecam_only=True loss_threshold=0.01 10 epochs training_supervised_ws
    # loss_threshold = 0.061 # best for small large use_finecam_only=False loss_threshold=0.061 10 epochs training_supervised_ws
    loss_threshold = 0.35
    use_finecam_only = False
    train_full_gt = False
    plot_result_masks = False
    epochs=4
    print('configurations:')
    print(f"vit model: {vit_model}")
    print(f"decoder_size: {decoder_size}")
    print(f"gt_ratio: {gt_ratio}")
    if train:
        print(f"seed: {seed}")
        print(f"loss_threshold: {loss_threshold}")
        print(f"use_finecam_only: {use_finecam_only}")
        print(f"train_full_gt: {train_full_gt}")
    print("=" * 50)
    # --- Transforms ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    if train:
        # print("Starting training process...")
        # print("=" * 50)

        # print("Splitting data into train and test sets...")
        # split_data.main(seed=seed)
        # print("=" * 50)

        # print("Training ViT model for image classification...")
        # train_ViT_classification.main(seed=seed, vit_model=vit_model, transform=transform)
        # print("=" * 50)

        # print("Building FineCAMs...")
        # build_fine_cams.main(vit_model=vit_model, transform=transform)
        # print("=" * 50)

        # print("Training CNN decoder...")
        # train_CNN_decoder.main(seed=seed, decoder_size=decoder_size, vit_model=vit_model, transform=transform)
        # print("=" * 50)

        print("Building masks...")
        build_masks.main(vit_model=vit_model, decoder_size=decoder_size, loss_threshold=loss_threshold, use_finecam_only=use_finecam_only, transform=transform, plot=False)
        print("=" * 50)

        if train_full_gt:
            print("Training supervised models...")
            print("Training Supervised Model with the Ground Truth Labels model...")
            train_supervised_gt.main(seed=seed)
            print("=" * 50)

        print("Training Supervised Model with the Pseudo Masks model...")
        train_supervised_ws.main(vit_model=vit_model, decoder_size=decoder_size, seed=seed, transform=transform, epochs=epochs, gt_ratio=gt_ratio)
        
        print("=" * 50)

    print("Testing results...")
    test_results.main(vit_model=vit_model, decoder_size=decoder_size, gt_ratio=gt_ratio, plot=plot_result_masks)
    print("=" * 50)

    print("All processes completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test the full segmentation pipeline.")
    parser.add_argument("--train", action="store_true", help="Enable training before testing.")
    args = parser.parse_args()

    main(train=args.train)
