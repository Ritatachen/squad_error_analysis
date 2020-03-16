from options import Options
from model import SQuAD
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

if __name__ == '__main__':
   opt = Options().parse()
   '''
   '''
   opt.model_type = "distilbert"
   # for more pretrain models, see https://github.com/huggingface/transformers/blob/b54ef78d0c30045bb3f9ecc8b178eab0dfdbeaec/docs/source/pretrained_models.rst
   opt.model_name_or_path = "distilbert-base-uncased-distilled-squad"
   opt.do_train = True
   opt.do_eval = True
   opt.do_lower_case = True
   opt.output_dir = "../models/wwm_uncased_finetuned_squad"
   opt.threads = 4

   m = SQuAD(opt)

   # do train
   global_step, tr_loss = m.train()
   m.save() # save the final model

   # do evaluation using the final model
   # m.load("path-to-checkpoint")
   result = m.evaluate(0)
   logger.info("Results: {}".format(result))