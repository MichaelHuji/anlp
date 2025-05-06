import argparse

def main():
    parser = argparse.ArgumentParser(description="Training and prediction script")

    parser.add_argument('--max_train_samples', type=int, help='Number of training samples')
    parser.add_argument('--max_eval_samples', type=int, help='Number of evaluation/validation samples')
    parser.add_argument('--max_predict_samples', type=int, help='Number of prediction samples')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--num_train_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--do_train', action='store_true', help='Flag to perform training')
    parser.add_argument('--do_predict', action='store_true', help='Flag to perform prediction')
    parser.add_argument('--model_path', type=str, help='Path to saved model for prediction')

    args = parser.parse_args()

    max_train_samples = args.max_train_samples
    max_eval_samples = args.max_eval_samples
    max_predict_samples = args.max_predict_samples
    lr = args.lr
    num_train_epochs = args.num_train_epochs
    batch_size = args.batch_size
    do_train = args.do_train
    do_predict = args.do_predict
    model_path = args.model_path


if __name__ == '__main__':
    main()


