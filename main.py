import argparse
import split_data
import train_ViT_classification
import build_fine_cams
import train_CNN_decoder
import build_masks
import train_supervised_gt
import train_supervised_ws
import test_results


def main(train: bool):
    seed = 24
    if train:
        print("Starting training process...")
        print("=" * 50)

        print("Splitting data into train and test sets...")
        split_data.main(seed=seed)
        print("=" * 50)

        print("Training ViT model for image classification...")
        train_ViT_classification.main(seed=seed)
        print("=" * 50)

        print("Building FineCAMs...")
        build_fine_cams.main()
        print("=" * 50)

        print("Training CNN decoder...")
        train_CNN_decoder.main(seed=seed)
        print("=" * 50)

        print("Building masks...")
        build_masks.main()
        print("=" * 50)

        print("Training supervised models...")
        print("Training Supervised Model with the Ground Truth Labels model...")
        train_supervised_gt.main(seed=seed)
        print("=" * 50)

        print("Training Supervised Model with the Pseudo Masks model...")
        train_supervised_ws.main(seed=seed)
        print("=" * 50)

    print("Testing results...")
    test_results.main()
    print("=" * 50)

    print("All processes completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test the full segmentation pipeline.")
    parser.add_argument("--train", action="store_true", help="Enable training before testing.")
    args = parser.parse_args()

    main(train=args.train)
