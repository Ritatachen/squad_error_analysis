import os
import torch
import timeit
import logging
import numpy as np
import pickle
from tqdm import tqdm, trange
from settings import MODEL_CLASSES
from utils import set_seed, to_list
from torch.utils.tensorboard import SummaryWriter
from data import prepare_dataset, create_dataloader
from transformers.data.processors.squad import SquadResult
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)


logger = logging.getLogger(__name__)


class SQuAD:
    def __init__(self, opt):
        # Load pretrained model and tokenizer
        config_class, self.model_class, self.tokenizer_class = MODEL_CLASSES[opt.model_type]

        self.config = config_class.from_pretrained(
            opt.config_name if opt.config_name else opt.model_name_or_path,
            cache_dir=opt.cache_dir if opt.cache_dir else None,
        )

        self.tokenizer = self.tokenizer_class.from_pretrained(
            opt.tokenizer_name if opt.tokenizer_name else opt.model_name_or_path,
            do_lower_case=opt.do_lower_case,
            cache_dir=opt.cache_dir if opt.cache_dir else None,
        )

        self.model = self.model_class.from_pretrained(
            opt.model_name_or_path,
            from_tf=bool(".ckpt" in opt.model_name_or_path),
            config=self.config,
            cache_dir=opt.cache_dir if opt.cache_dir else None,
        )

        self.model.to(opt.device)
        logger.info("Training/evaluation parameters %s", opt)

        # Load dataset and dataloader
        self.train_dataset = prepare_dataset(opt, self.tokenizer, evaluate=False, output_examples=False)
        self.train_dataloader = create_dataloader(self.train_dataset, batchsize=opt.train_batch_size, evaluate=False)

        self.eval_dataset, self.examples, self.features = prepare_dataset(opt, self.tokenizer, evaluate=True, output_examples=True)
        self.eval_dataloader = create_dataloader(self.eval_dataset, batchsize=opt.eval_batch_size, evaluate=True)

        self.opt = opt

    def train(self):
        tb_writer = SummaryWriter()
        t_total = len(self.train_dataloader) // self.opt.gradient_accumulation_steps * self.opt.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.opt.weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.opt.learning_rate, eps=self.opt.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.opt.warmup_steps, num_training_steps=t_total
        )

        # Check if saved optimizer or scheduler states exist
        if os.path.isfile(os.path.join(self.opt.model_name_or_path, "optimizer.pt")) and os.path.isfile(
                os.path.join(self.opt.model_name_or_path, "scheduler.pt")):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(self.opt.model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(self.opt.model_name_or_path, "scheduler.pt")))

        # Train !
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.opt.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", self.opt.train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            self.opt.train_batch_size
            * self.opt.gradient_accumulation_steps
        )
        logger.info("  Gradient Accumulation steps = %d", self.opt.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 1
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if os.path.exists(self.opt.model_name_or_path):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = self.opt.model_name_or_path.split("-")[-1].split("/")[0]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (len(self.train_dataloader) // self.opt.gradient_accumulation_steps)
                steps_trained_in_current_epoch = global_step % (
                            len(self.train_dataloader) // self.opt.gradient_accumulation_steps)

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                logger.info("  Starting fine-tuning.")

        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        set_seed(self.opt)

        for epoch in range(epochs_trained, self.opt.num_train_epochs):
            logging.info("  Start training epoch %d", epoch)
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration", leave=True, total=len(self.train_dataloader))):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                self.model.train()
                batch = tuple(t.to(self.opt.device) for t in batch)

                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "start_positions": batch[3],
                    "end_positions": batch[4],
                }

                if self.opt.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                    del inputs["token_type_ids"]

                if self.opt.model_type in ["xlnet", "xlm"]:
                    inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
                    inputs.update({"is_impossible": batch[7]})
                    if hasattr(self.model, "config") and hasattr(self.model.config, "lang2id"):
                        inputs.update(
                            {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * self.opt.lang_id).to(self.opt.device)}
                        )

                outputs = self.model(**inputs)
                # model outputs are always tuple in transformers (see doc)
                loss = outputs[0]

                if self.opt.gradient_accumulation_steps > 1:
                    loss = loss / self.opt.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.opt.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    # Log metrics
                    if self.opt.logging_steps > 0 and global_step % self.opt.logging_steps == 0:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        if self.opt.evaluate_during_training:
                            results = self.evaluate() # need self.opt, self.model, self.tokenizer
                            for key, value in results.items():
                                tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar("loss", (tr_loss - logging_loss) / self.opt.logging_steps, global_step)
                        logging_loss = tr_loss

                    # Save model checkpoint
                    if self.opt.save_steps > 0 and global_step % self.opt.save_steps == 0:
                        output_dir = os.path.join(self.opt.output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # Take care of distributed/parallel training
                        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
                        model_to_save.save_pretrained(output_dir)
                        self.tokenizer.save_pretrained(output_dir)

                        torch.save(self.opt, os.path.join(output_dir, "training_self.opt.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

        tb_writer.close()

        return global_step, tr_loss / global_step

    def evaluate(self, prefix=''):
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(self.eval_dataset))
        logger.info("  Batch size = %d", self.opt.eval_batch_size)

        all_results = []
        start_time = timeit.default_timer()

        for batch in tqdm(self.eval_dataloader, desc="Evaluating", leave=True):
            self.model.eval()
            batch = tuple(t.to(self.opt.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                if self.opt.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                    del inputs["token_type_ids"]

                example_indices = batch[3]

                # XLNet and XLM use more arguments for their predictions
                if self.opt.model_type in ["xlnet", "xlm"]:
                    inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                    # for lang_id-sensitive xlm models
                    if hasattr(self.model, "config") and hasattr(self.model.config, "lang2id"):
                        inputs.update(
                            {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * self.opt.lang_id).to(self.opt.device)}
                        )

                outputs = self.model(**inputs)
                if self.model.config.output_attentions:
                    attentions = outputs[2][-4:]
                    outputs = outputs[:2]

            for i, example_index in enumerate(example_indices):
                eval_feature = self.features[example_index.item()]
                if self.model.config.output_attentions:
                    unique_id = int(eval_feature.unique_id)
                    attention = np.array([attention[i].cpu().numpy() for attention in attentions])
                    tokens = self.features[example_index].tokens
                    qas_idx = self.examples[example_index].qas_id
                    attention_output = {
                        'attention': attention,
                        'tokens': tokens,
                        'qas_idx': qas_idx
                    }
                    pickle.dump(attention_output, open('{}.pkl'.format(qas_idx), 'wb'))
                output = [to_list(output[i]) for output in outputs]

                # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
                # models only use two.
                if len(output) >= 5:
                    start_logits = output[0]
                    start_top_index = output[1]
                    end_logits = output[2]
                    end_top_index = output[3]
                    cls_logits = output[4]

                    result = SquadResult(
                        unique_id,
                        start_logits,
                        end_logits,
                        start_top_index=start_top_index,
                        end_top_index=end_top_index,
                        cls_logits=cls_logits,
                    )

                else:
                    start_logits, end_logits = output
                    result = SquadResult(unique_id, start_logits, end_logits)

                all_results.append(result)

        evalTime = timeit.default_timer() - start_time
        logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(self.eval_dataset))

        # Compute predictions
        output_prediction_file = os.path.join(self.opt.output_dir, "predictions_{}.json".format(prefix))
        output_nbest_file = os.path.join(self.opt.output_dir, "nbest_predictions_{}.json".format(prefix))

        output_null_log_odds_file = os.path.join(self.opt.output_dir, "null_odds_{}.json".format(prefix))

        # XLNet and XLM use a more complex post-processing procedure
        if self.opt.model_type in ["xlnet", "xlm"]:
            start_n_top = self.model.config.start_n_top if hasattr(self.model, "config") else self.model.module.config.start_n_top
            end_n_top = self.model.config.end_n_top if hasattr(self.model, "config") else self.model.module.config.end_n_top

            predictions = compute_predictions_log_probs(
                self.examples,
                self.features,
                all_results,
                self.opt.n_best_size,
                self.opt.max_answer_length,
                output_prediction_file,
                output_nbest_file,
                output_null_log_odds_file,
                start_n_top,
                end_n_top,
                True,
                self.tokenizer,
                self.opt.verbose_logging,
            )
        else:
            predictions = compute_predictions_logits(
                self.examples,
                self.features,
                all_results,
                self.opt.n_best_size,
                self.opt.max_answer_length,
                self.opt.do_lower_case,
                output_prediction_file,
                output_nbest_file,
                output_null_log_odds_file,
                self.opt.verbose_logging,
                True,
                self.opt.null_score_diff_threshold,
                self.tokenizer,
            )


        # Compute the F1 and exact scores.
        results = squad_evaluate(self.examples, predictions)
        return results

    def save(self):
        # should call after train
        if not os.path.exists(self.opt.output_dir):
            os.makedirs(self.opt.output_dir)

        logger.info("Saving model checkpoint to %s", self.opt.output_dir)
        self.model.save_pretrained(self.opt.output_dir)
        self.tokenizer.save_pretrained(self.opt.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.opt, os.path.join(self.opt.output_dir, "training_self.opt.bin"))

    def load(self, checkpoint):
        config_class, self.model_class, self.tokenizer_class = MODEL_CLASSES[self.opt.model_type]
        self.model = self.model_class.from_pretrained(checkpoint)  # , force_download=True)
        self.tokenizer = self.tokenizer_class.from_pretrained(checkpoint, do_lower_case=self.opt.do_lower_case)
        self.model.to(self.opt.device)
