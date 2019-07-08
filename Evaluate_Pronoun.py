import os

import tensorflow as tf
import pronoun_coref_kg_model as cm
import util

if __name__ == "__main__":
  config = util.initialize_from_env()
  model = cm.PronounCorefKGModel(config, 'test')
  os.environ["CUDA_VISIBLE_DEVICES"] = "1"
  with tf.Session() as session:
    model.restore(session)
    model.test(session, official_stdout=True)