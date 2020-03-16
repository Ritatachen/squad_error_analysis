import argparse
from settings import MODEL_CLASSES, ALL_MODELS

class Options():

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument(
            "--model_type",
            default=None,
            type=str,
            # required=True,
            help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
        )
        self.parser.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            # required=True,
            help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
        )
        self.parser.add_argument(
            "--output_dir",
            default=None,
            type=str,
            # required=True,
            help="The output directory where the model checkpoints and predictions will be written.",
        )

        # Other parameters
        self.parser.add_argument(
            "--data_dir",
            default=None,
            type=str,
            help="The input data dir. Should contain the .json files for the task."
                 + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
        )
        self.parser.add_argument(
            "--train_file",
            default=None,
            type=str,
            help="The input training file. If a data dir is specified, will look for the file there"
                 + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
        )
        self.parser.add_argument(
            "--predict_file",
            default=None,
            type=str,
            help="The input evaluation file. If a data dir is specified, will look for the file there"
                 + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
        )
        self.parser.add_argument(
            "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
        )
        self.parser.add_argument(
            "--tokenizer_name",
            default="",
            type=str,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )
        self.parser.add_argument(
            "--cache_dir",
            default="",
            type=str,
            help="Where do you want to store the pre-trained models downloaded from s3",
        )

        self.parser.add_argument(
            "--null_score_diff_threshold",
            type=float,
            default=0.0,
            help="If null_score - best_non_null is greater than the threshold predict null.",
        )

        self.parser.add_argument(
            "--max_seq_length",
            default=384,
            type=int,
            help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                 "longer than this will be truncated, and sequences shorter than this will be padded.",
        )
        self.parser.add_argument(
            "--doc_stride",
            default=128,
            type=int,
            help="When splitting up a long document into chunks, how much stride to take between chunks.",
        )
        self.parser.add_argument(
            "--max_query_length",
            default=64,
            type=int,
            help="The maximum number of tokens for the question. Questions longer than this will "
                 "be truncated to this length.",
        )
        self.parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
        self.parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
        self.parser.add_argument(
            "--evaluate_during_training", action="store_true",
            help="Run evaluation during training at each logging step."
        )
        self.parser.add_argument(
            "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
        )

        self.parser.add_argument("--train_batch_size", default=8, type=int,  help="Batch size for training.")
        self.parser.add_argument("--eval_batch_size", default=8, type=int, help="Batch size for evaluation.")
        self.parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
        self.parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.",
        )
        self.parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        self.parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        self.parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
        self.parser.add_argument(
            "--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform."
        )

        self.parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
        self.parser.add_argument(
            "--n_best_size",
            default=20,
            type=int,
            help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
        )
        self.parser.add_argument(
            "--max_answer_length",
            default=30,
            type=int,
            help="The maximum length of an answer that can be generated. This is needed because the start "
                 "and end predictions are not conditioned on one another.",
        )
        self.parser.add_argument(
            "--verbose_logging",
            action="store_true",
            help="If true, all of the warnings related to data processing will be printed. "
                 "A number of warnings are expected for a normal SQuAD evaluation.",
        )
        self.parser.add_argument(
            "--lang_id",
            default=0,
            type=int,
            help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
        )

        self.parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
        self.parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
        self.parser.add_argument(
            "--eval_all_checkpoints",
            action="store_true",
            help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
        )
        self.parser.add_argument("--cpu", action="store_true", help="Whether not to use CUDA when available")
        self.parser.add_argument(
            "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
        )
        self.parser.add_argument(
            "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
        )
        self.parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

        self.parser.add_argument("--threads", type=int, default=1,
                            help="multiple threads for converting example to features")

        self.opt = None

    def parse(self):
        """ Parse Arguments.
        """
        self.opt = self.parser.parse_args()
        self.opt.device = "cpu" if self.opt.cpu else "cuda"
        return self.opt



#define argparse for experiment