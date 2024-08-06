import warnings
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import logging as hf_logging
import logging
import os
logging.basicConfig(level=logging.ERROR)
hf_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)
def get_model(args):

    return AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.num_labels)

