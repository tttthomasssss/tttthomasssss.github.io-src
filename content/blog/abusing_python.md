Title: (Ab)using Python for Better Debbuging of Pre-trained Pytorch Models
Date: 2019-05-11 09:54
Category: Blog
Tags: python, machine learning, data science, neural networks, pytorch
Slug: abusing-python-for-better-debugging-of-pre-trained-pytorch-models
Authors: Thomas
Summary: It can be useful to (ab)use python's dynamic object model to add functions to existing objects in order to better interrogate and debug a pre-trained pytorch model.

Building bigger models to achieve better performance is a common theme in current NLP research papers, with many authors releasing their code and/or some pre-trained models. Having access to a pre-trained model is great, because its almost always infeasible to train a model from scratch because one might not have access to the appropriate computing resources or the same data that the original authors used for training their model.

Thus, we're often re-using and running existing pre-trained models and it becomes increasingly important to understand what they learn and to interrogate any intermediate representations they create in order to get a better feel for whats inside the black box.

Luckily sharing pre-trained models is relatively easy with current tools such as pytorch or tensorflow. In this post, I'll briefly discuss how its possible to debug an existing pre-trained model by dynamically adding methods to it.

In pytorch, a pre-trained model is just a python object, so its possible to overwrite and modify existing instance methods by whatever we want to do. 

Lets start with an actual example. For a [recent paper]() I have pre-trained a relatively straightforward bidirectional LSTM with max-pooling on the [SNLI dataset](). It generally follows the [InferSent]() model architecture, but I use fewer hidden units, thereby trading off some points of accuracy for faster training. Essentially I wasn't interested in +/- 2 points of improvement on a dataset, but whether the model works _in principle_ on my dataset. The pre-trained model achieves 0.83 accuracy on the SNLI dev set and 0.82 on the SNLI test set. Note that the pre-trained model is around 2.5gb, so I can't share it on this site, but feel free to [get in touch](../pages/contact) if you need access to the model.

The code for the model is relatively straightforward (Note, the code for `EmbeddingLayer`, `FeedForward` and `LSTMEncoder` currently resides in a private bitbucket repository alongside the rest of my research code, please [contact me](../pages/contact) if you would like to get some of the code):

```python
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from semantx.models.pytorch_layers import EmbeddingLayer # wraps an Embedding object
from semantx.models.pytorch_model_factory import FeedForward # wraps some basic feedforward layer functionality
from semantx.models.pytorch_model_factory import LSTMEncoder # wraps a basic nn.LSTM layer


class NLIEncoderFeedforward(nn.Module):
	def __init__(self, embedding_config, encoder_config, aggregation_layer, feedforward_config, pooling_layer,
				 conditional_encoding=False):
		super(NLIEncoderFeedforward, self).__init__()
		self.embedding = EmbeddingLayer(**embedding_config)
		self.encoder = LSTMEncoder(**encoder_config)
		self.pooling = pooling_layer
		self.aggregation = aggregation_layer
		self.feedforward = FeedForward(**feedforward_config)
		self.conditional_encoding = conditional_encoding

	def forward(self, batch):
		# Embed sequences
		premise = self.embedding(batch.premise)
		hypothesis = self.embedding(batch.hypothesis)

		# Pack embedded sequences
		premise = pack_padded_sequence(premise, batch.premise_lengths, batch_first=True)
		hypothesis = pack_padded_sequence(hypothesis, batch.hypothesis_lengths, batch_first=True)

		# Encode premise & hypothesis
		premise, (hidden, cell_state) = self.encoder(premise, curr_batch_size=len(batch))
		if (self.conditional_encoding):
			hypothesis, *_ = self.encoder(hypothesis, curr_batch_size=len(batch), hidden=hidden, cell_state=cell_state)
		else:
			hypothesis, *_ = self.encoder(hypothesis, curr_batch_size=len(batch))

		# The packed sequences are length-sorted, so we need to return them to their original ordering
		premise = premise[batch.premise_orig_idx, :, :]
		hypothesis = hypothesis[batch.hypothesis_orig_idx, :, :]

		# Pool premise & hypothesis
		premise_emb = self.pooling(premise, batch_first=True)
		hypothesis_emb = self.pooling(hypothesis, batch_first=True)

		# Combine the premise & hypothesis representations
		ff_input = self.aggregation(premise_emb, hypothesis_emb)

		# Feedforward classification with the embedded sequences
		scores = self.feedforward(ff_input)

		return scores
```

The model first uses an LSTM to encode the premise and the hypothesis, performs some pooling on the encoded sequences and subsequently aggregates the two representations into a single vector before passing it on to a standard feedforward network that returns the class predictions. Its all nice and good if all we are interested in is the final predictions. However, sometimes we might want to look into the model in order figure out whats going on in intermediate steps, i.e. whats the benefit of using a max pooling layer? Does it learn anything interesting? If we look at the neighbourhood before and after applying pooling, what do we get? Whats the impact of different aggregation functions on the sentence space?

In order to answer any of these questions we could either get all of that code, download SNLI, try to preprocess it the same way that I did, hope for the best that I have listed all hyperparameters in the paper and then train the model. 

But thats just a big waste of resources, time and sanity. Its much easier to [contact me](../pages/contact) and get the actual pre-trained model as well as the vectoriser that converts any input text to embedding indices that go into the model. Then we can just load the model and start hacking it.

```python
import joblib
import torch

model = torch.load('path/to/the/pretrained/model')
vec = joblib.load('path/to/the/vectoriser')
```

Now, lets create two functions that return the premise and the hypothesis before and after the pooling operation, as well as a function that returns aggregated representation (luckily we have the code of the model just above, so all we really need to do is some method definitions with some copy/paste).

```python
def get_representations_before_pooling(self, batch):
    # Embed sequences
	premise = self.embedding(batch.premise)
	hypothesis = self.embedding(batch.hypothesis)

	# Pack embedded sequences
	premise = pack_padded_sequence(premise, batch.premise_lengths, batch_first=True)
	hypothesis = pack_padded_sequence(hypothesis, batch.hypothesis_lengths, batch_first=True)

	# Encode premise & hypothesis
	premise, (hidden, cell_state) = self.encoder(premise, curr_batch_size=len(batch))
	if (self.conditional_encoding):
		hypothesis, *_ = self.encoder(hypothesis, curr_batch_size=len(batch), hidden=hidden, cell_state=cell_state)
	else:
		hypothesis, *_ = self.encoder(hypothesis, curr_batch_size=len(batch))

	# The packed sequences are length-sorted, so we need to return them to their original ordering
	premise = premise[batch.premise_orig_idx, :, :]
	hypothesis = hypothesis[batch.hypothesis_orig_idx, :, :]
	
	return premise, hypothesis
	

def get_representations_after_pooling(self, batch):
	premise, hypothesis = self.get_representations_before_pooling(batch)
	
	# Pool premise & hypothesis
	premise_emb = self.pooling(premise, batch_first=True)
	hypothesis_emb = self.pooling(hypothesis, batch_first=True)
	
	return premise_emb, hypothesis_emb
	

def get_aggregated_representations(self, batch):
	premise_emb, hypothesis_emb = self.get_representations_after_pooling(batch)
	
	# Combine the premise & hypothesis representations
	aggr = self.aggregation(premise_emb, hypothesis_emb)
	
	return aggr
``` 

Easy peasy, we basically just copy pasted the relevant functionality from the forward function into smaller chunks. Now the last missing bit is dynamically binding these functions to the model object we've loaded above. While this is arguably a bit hacky, its well defined within python by using `types`.

```python
import types

model.get_representations_before_pooling = types.MethodType(get_representations_before_pooling, model)
print(model.get_representations_before_pooling)

# Outputs
# <bound method get_representations_before_pooling of NLIEncoderFeedforward(
#  (embedding): EmbeddingLayer(
#    (word_embeddings): Embedding(2196010, 300)
#  )
#  (encoder): LSTMEncoder(
#    (lstm): LSTM(300, 300, bidirectional=True)
#  )
#  (pooling): BidirectionalMaxPoolingLayer()
#  (aggregation): NLIAggregationLayerBalazs2017()
#  (feedforward): FeedForward(
#    (forward_layer_1): FeedforwardLayer(
#      (feedforward): Sequential(
#        (0): Linear(in_features=2400, out_features=2400, bias=True)
#        (1): ReLU()
#        (2): Dropout(p=0.0)
#      )
#    )
#    (output_layer): Linear(in_features=2400, out_features=3, bias=True)
#  )
#)>
# We just bound the above defined function to a _single_ object instance and we can now just this method as if its always been there! Lets also add the other functions

model.get_representations_after_pooling = types.MethodType(get_representations_after_pooling, model)
model.get_aggregated_representations = types.MethodType(get_aggregated_representations, model)
```

Cool stuff! We can now pass in some example sentences with which we'd like to interrogate the model behaviour and thereby gain some more insight into what the model has learnt! For the sake of simplicity, lets pass in two premise-hypothesis pairs and compare the cosines of their representations before and after pooling. For the representation before pooling, we are going to choose the final hidden state.

The two pairs we're passing in are:

* _Five men are playing frisbee on the beach_ **[ENTAILS]** _Some people are playing at the seafront_
* _Five men are playing frisbee on the beach_ **[NOT ENTAILS]** _A group of women is crossing a busy road_

Lets vectorise them and pass them to our new functions (**Note:** this type of interrogation hinges on knowledge of the preprocessing pipeline.)!

```python
from scipy.spatial.distance import cosine
import numpy as np

# Create premises and hypotheses
premises = ['Five men are playing frisbee on the beach', 'Five men are playing frisbee on the beach']
hypotheses = ['Some people are playing at the seafront', 'A group of women is crossing a busy road']

# Check how the SNLI labels are mapped
print(vec.label_encoder.inverse_transform(np.array([0, 1, 2])))

# Output
# array(['contradiction', 'entailment', 'neutral'], dtype='<U13')

# So the first example is an entailment, thus the label index is 1, the second one is a contradiction, hence we use 0
y = np.array([1, 0])

batch = vec.batch_transform_nli_padded_sorted(raw_document_1=premises, raw_documents_2=hypotheses, y=y)
```

Okidoki, now we have everything together to do some advanced model debugging :).

```python
prem_b4_pool, hypo_b4_pool = model.get_representations_before_pooling(batch)

prem_after_pool, hypo_after_pool = model.get_representations_after_pooling(batch)

# Print their shapes
print(prem_b4_pool.shape, prem_after_pool.shape)

# Output
# torch.Size([2, 8, 600]) torch.Size([2, 600])

# For the representation before pooling we said we were just going to use the final state
prem_b4_pool = prem_b4_pool[:, -1, :].squeeze()
hypo_b4_pool = hypo_b4_pool[:, -1, :].squeeze()

# And now we throw everything into numpy arrays, because we like numpy arrays
P_b4_pool = prem_b4_pool.detach().numpy()
H_b4_pool = hypo_b4_pool.detach().numpy()
P_after_pool = prem_after_pool.detach().numpy()
H_after_pool = hypo_after_pool.detach().numpy()

# So, lets look at the cosines between the contradicting case before and after pooling (remember, in scipy its cosine distance, so cosine similarity = 1 - distance)
print(1 - cosine(P_b4_pool[1], H_b4_pool[1]))

# Output
# 0.3818543255329132

print(1 - cosine(P_after_pool[1], H_after_pool[1]))

# Output
# 0.4067332446575165

# So pooling has made the two representations slightly more similar. How about the entailment case?
print(1 - cosine(P_b4_pool[0], H_b4_pool[0]))

# Output
# 0.6774758696556091

print(1 - cosine(P_after_pool[1], H_after_pool[1]))

# Output
# 0.6271609663963318
```

Interestingly, max pooling made the contradiction pair _more_ similar, whereas it made the entailment pair _less_ similar (however in absolute numbers, the entailment pair is much more similar than the contradiction pair).

Of course this quick example is not terribly informative, however we are now in a position to calculate the nearest neighbours of some cases and measure the effect of pooling by looking how the neighbourhood of individual sentences changes.