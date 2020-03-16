import os
import torch
import logging
from transformers import squad_convert_examples_to_features
from transformers.data.processors.squad import SquadV2Processor
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
logger = logging.getLogger(__name__)

def prepare_dataset(args, tokenizer, evaluate=False, output_examples=False):

    # tensorflow_datasets does not handle SQuAD2.0, we can not use tensorflow_datasets to load data
    if not args.data_dir and ((evaluate and not args.predict_file) or (not evaluate and not args.train_file)):
        data_dir, train_file, predict_file = "../data/SQuAD2", "train-v2.0.json", "dev-v2.0.json"
    else:
        data_dir, train_file, predict_file = args.data_dir, args.train_file, args.predict_file

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(os.path.join(data_dir, train_file)):
        os.system("curl -o {} https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
                  .format(os.path.join(data_dir, train_file)))
    if not os.path.exists(os.path.join(data_dir, predict_file)):
        os.system("curl -o {} https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
                  .format(os.path.join(data_dir, predict_file)))

    # Load data features from cache or dataset file
    input_dir = data_dir
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        processor = SquadV2Processor()
        if evaluate:
            examples = processor.get_dev_examples(data_dir, filename=predict_file)
        else:
            examples = processor.get_train_examples(data_dir, filename=train_file)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=args.threads,
        )

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)
    if output_examples:
        return dataset, examples, features
    return dataset


def create_dataloader(dataset, batchsize, evaluate=False):
    if evaluate:
        sampler = SequentialSampler(dataset)
    else:
        sampler = RandomSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=batchsize)


if __name__ == '__main__':
    from options import Options, MODEL_CLASSES
    opt = Options().parse()
    opt.model_type = "bert"
    opt.model_name_or_path = "bert-large-uncased-whole-word-masking"
    opt.do_train = True
    opt.do_eval = True
    opt.do_lower_case = True
    opt.output_dir = "../models/wwm_uncased_finetuned_squad"
    opt.threads = 4
    config_class, model_class, tokenizer_class = MODEL_CLASSES[opt.model_type]
    tokenizer = tokenizer_class.from_pretrained(
        opt.tokenizer_name if opt.tokenizer_name else opt.model_name_or_path,
        do_lower_case=opt.do_lower_case,
        cache_dir=opt.cache_dir if opt.cache_dir else None,
    )
    train_dataset = prepare_dataset(opt, tokenizer, evaluate=False, output_examples=False)
    pass
