Title: Publications

I'm trying to keep this page up-to-date as much as possible, but if you don't find what you're looking for either check my [Google Scholar](https://scholar.google.at/citations?user=1YCMKBgAAAAJ&hl=en) page or [contact](./contact) me directly.

Alternatively, the new ACL Anthology <a href="https://www.aclweb.org/anthology/people/t/thomas-kober/" target="_blank">people page</a> also has a complete list of all papers published in any of its venues. 


### 2019

##### Temporal and Aspectual Entailment

_Thomas Kober, Sander Bijl de Vroe & Mark Steedman_

Inferences regarding Jane’s arrival in London from predications such as _Jane is going to London_ or _Jane has gone to London_ depend on tense and aspect of the predications. Tense determines the temporal location of the predication in the past, present or future of the time of utterance. The aspectual auxiliaries on the other hand specify the internal constituency of the event, i.e. whether the event of _going to London_ is completed and whether its consequences hold at that time or not.

While tense and aspect are among the most important factors for determining natural language inference, there has been very little work to show whether modern NLP models capture these semantic concepts. In this paper we propose a novel entailment dataset and analyse the ability of a range of recently proposed NLP models to perform inference on temporal predications. We show that the models encode a substantial amount of morphosyntactic information relating to tense and aspect, but fail to model inferences that require reasoning with these semantic properties.

<a href="{static}/pdfs/papers/TemporalAndAspectualEntailment_IWCS.pdf" target="_blank" class="label label-primary">paper</a> <a href="{static}/bibtex/Kober_2019.bib" target="_blank" class="label label-info">bibtex</a> <a href="{static}/pdfs/slides/IWCS2019.pdf" target="_blank" class="label label-default">slides</a> <a href="https://github.com/tttthomasssss/iwcs2019" target="_blank" class="label label-success">code</a> <a href="{static}/datasets/TEA.txt" target="_blank" class="label label-danger">TEA Dataset</a> <a href="{static}/datasets/aux_verb_agreement.txt" target="_blank" class="label label-danger">Auxiliary-Verb Agreement Dataset</a> <a href="{static}/datasets/translation_operation.txt" target="_blank" class="label label-danger">Translation Operation Dataset</a> 

### 2018

##### Inferring Unobserved Co-occurrence Events in Anchored Packed Trees

_Thomas Kober_

Anchored Packed Trees (APTs) are a novel approach to distributional semantics that takes distributional composition to be a process of lexeme contextualisation. A lexeme’s meaning, characterised as knowledge concerning co-occurrences involving that lexeme, is represented with a higher-order dependency-typed structure (the APT) where paths associated with higher-order dependencies connect vertices associated with weighted lexeme multisets. The central innovation in the compositional theory is that the APT’s type structure enables the precise alignment of the semantic representation of each of the lexemes being composed.

Like other count-based distributional spaces, however, Anchored Packed Trees are prone to considerable data sparsity, caused by not observing all plausible co-occurrences in the given data. This problem is amplified for models like APTs, that take the grammatical type of a co-occurrence into account. This results in a very sparse distributional space, requiring a mechanism for inferring missing knowledge. Most methods face this challenge in ways that render the resulting word representations uninterpretable, with the consequence that distributional composition becomes difficult to model and reason about.

In this thesis, I will present a practical evaluation of the APT theory, including a large-scale hyperparameter sensitivity study and a characterisation of the distributional space that APTs give rise to. Based on the empirical analysis, the impact of the problem of data sparsity is investigated. In order to address the data sparsity challenge and retain the interpretability of the model, I explore an alternative algorithm — distributional inference — for improving elementary representations. The algorithm involves explicitly inferring unobserved co-occurrence events by leveraging the distributional neighbourhood of the semantic space. I then leverage the rich type structure in APTs and propose a generalisation of the distributional inference algorithm. I empirically show that distributional inference improves elementary word representations and is especially beneficial when combined with an intersective composition function, which is due to the complementary nature of inference and composition. Lastly, I qualitatively analyse the proposed algorithms in order to characterise the knowledge that they are able to infer, as well as their impact on the distributional APT space.

<a href="{static}/pdfs/papers/thesis.pdf" target="_blank" class="label label-primary">phd thesis</a> <a href="{static}/bibtex/Kober_2018.bib" target="_blank" class="label label-info">bibtex</a>

### 2017

##### Improving Semantic Composition with Offset Inference

_Thomas Kober, Julie Weeds, Jeremy Reffin & David Weir_

Count-based distributional semantic models suffer from sparsity due to unobserved but plausible co-occurrences in any text collection. This problem is amplified for models like Anchored Packed Trees (APTs), that take the grammatical type of a co-occurrence into account. We therefore introduce a novel form of distributional inference that exploits the rich type structure in APTs and infers missing data by the same mechanism that is used for semantic composition.

<a href="{static}/pdfs/papers/OffsetInference_ACL.pdf" target="_blank" class="label label-primary">paper</a> <a href="{static}/bibtex/Kober_2017b.bib" target="_blank" class="label label-info">bibtex</a> <a href="{static}/pdfs/posters/OffsetInference_Poster.pdf" target="_blank" class="label label-default">poster</a> <a href="https://github.com/tttthomasssss/apt-toolkit" target="_blank" class="label label-success">code</a>
<br /><br />

##### When a Red Herring is Not a Red Herring: Using Compositional Methods to Detect Non-Compositional Phrases

_Julie Weeds, Thomas Kober, Jeremy Reffin & David Weir_

Non-compositional phrases such as red herring and weakly compositional phrases such as spelling bee are an integral part of natural language (Sag et al., 2002). They are also the phrases that are difficult, or even impossible, for good compositional distributional models of semantics. Compositionality detection therefore provides a good testbed for compositional methods. We compare an integrated compositional distributional approach, using sparse high dimensional representations, with the ad-hoc compositional approach of applying simple composition operations to state-of-the-art neural embeddings.

<a href="{static}/pdfs/papers/RedHerring_EACL.pdf" target="_blank" class="label label-primary">paper</a> <a href="{static}/bibtex/Weeds_2017.bib" target="_blank" class="label label-info">bibtex</a>
<br /><br />

##### One Representation per Word - Does it make _Sense_ for Composition?

_Thomas Kober, Julie Weeds, John Wilkie, Jeremy Reffin & David Weir_

In this paper, we investigate whether an _a priori_ disambiguation of word senses is strictly necessary or whether the meaning of a word in context can be disambiguated through composition alone. We evaluate the performance of off-the-shelf single-vector and multi-sense vector models on a benchmark phrase similarity task and a novel task for word-sense discrimination. We find that single-sense vector models perform as well or better than multi-sense vector models despite arguably less clean elementary representations. Our findings furthermore show that simple composition functions such as pointwise addition are able to recover sense specific information from a single-sense vector model remarkably well.

<a href="{static}/pdfs/papers/Composition_SENSE.pdf" target="_blank" class="label label-primary">paper</a> <a href="{static}/bibtex/Kober_2017.bib" target="_blank" class="label label-info">bibtex</a> <a href="{static}/pdfs/slides/sense2017_talk.pdf" target="_blank" class="label label-default">slides</a> <a href="https://github.com/tttthomasssss/sense2017" target="_blank" class="label label-success">code</a>

### 2016

##### A critique of word similarity as a method for evaluating distributional semantic models

_Miroslav Batchkarov, Thomas Kober, Jeremy Reffin, Julie Weeds & David Weir_

This paper aims to re-think the role of the word similarity task in distributional semantics research. We argue while it is a valuable tool, it should be used with care because it provides only an approximate measure of the quality of a distributional model. Word similarity evaluations assume there exists a single notion of similarity that is independent of a particular application. Further, the small size and low inter-annotator agreement of existing data sets makes it challenging to find significant differences between models.

<a href="{static}/pdfs/papers/WordSimCritique_REPEVAL.pdf" target="_blank" class="label label-primary">paper</a> <a href="{static}/bibtex/Batchkarov_2016.bib" target="_blank" class="label label-info">bibtex</a> <a href="https://github.com/tttthomasssss/repeval2016" target="_blank" class="label label-success">code</a>
<br /><br />

##### Improving Sparse Word Representations with Distributional Inference for Semantic Composition

_Thomas Kober, Julie Weeds, Jeremy Reffin & David Weir_

Distributional models are derived from co-occurrences in a corpus, where only a small proportion of all possible plausible co-occurrences will be observed. This results in a very sparse vector space, requiring a mechanism for inferring missing knowledge. Most methods face this challenge in ways that render the resulting word representations uninterpretable, with the consequence that semantic composition becomes hard to model. In this paper we explore an alternative which involves explicitly inferring unobserved co-occurrences using the distributional neighbourhood. We show that distributional inference improves sparse word representations on several word similarity benchmarks and demonstrate that our model is competitive with the state-of-the-art for adjective-noun, noun-noun and verb-object compositions while being fully interpretable.

<a href="{static}/pdfs/papers/ImprovingSparseWordREpsWithDI_EMNLP.pdf" target="_blank" class="label label-primary">paper</a> <a href="{static}/bibtex/Kober_2016.bib" target="_blank" class="label label-info">bibtex</a> <a href="{static}/pdfs/posters/EMNLP2016_poster.pdf" target="_blank" class="label label-default">poster</a> <a href="https://github.com/tttthomasssss/apt-toolkit" target="_blank" class="label label-success">code</a>
<br /><br />

##### Aligning Packed Dependency Trees: A Theory of Composition for Distributional Semantics

_David Weir, Julie Weeds, Jeremy Reffin & Thomas Kober_

We present a new framework for compositional distributional semantics in which the distributional contexts of lexemes are expressed in terms of anchored packed dependency trees. We show that these structures have the potential to capture the full sentential contexts of a lexeme and provide a uniform basis for the composition of distributional knowledge in a way that captures both mutual disambiguation and generalization.

<a href="{static}/pdfs/papers/APT_CL_special_issue.pdf" target="_blank" class="label label-primary">paper</a> <a href="{static}/bibtex/Weir_2016.bib" target="_blank" class="label label-info">bibtex</a>


### 2015

##### Optimising Agile Social Media Analysis

_Thomas Kober & David Weir_

Agile social media analysis involves building bespoke, one-off classification pipelines tailored to the analysis of specific datasets. In this study we investigate how the DUALIST architecture can be optimised for agile social media analysis. We evaluate several semi-supervised learning algorithms in conjunction with a Naıve Bayes model, and show how these modifications can improve the performance of bespoke classifiers for a variety of tasks on a large range of datasets.

<a href="{static}/pdfs/papers/NB_SemiSupervised_WASSA.pdf" target="_blank" class="label label-primary">paper</a> <a href="{static}/bibtex/Kober_2015.bib" target="_blank" class="label label-info">bibtex</a> <a href="{static}/pdfs/slides/WASSA2015_Presentation.pdf" target="_blank" class="label label-default">slides</a> <a href="https://github.com/tttthomasssss/common_utils/blob/master/classifiers/naive_bayes.py" target="_blank" class="label label-success">code</a>

### 2014

##### Scaling Semi-Supervised Multinomial Naive Bayes

_Thomas Kober_

The project originated during a summer internship in the Text Analytics Group at the University of Sussex. My aim for the internship was to implement a Maximum Entropy classifier (MaxEnt) for an active learning tool called DUALIST, to replace the Multinomial Naıve Bayes classifier used in the application, as MaxEnt was found to be superior to Naıve Bayes in previous studies, i.e. Nigam (1999). However, I found that for the purpose of social media analysis on Twitter in the given active learning scenario, the MaxEnt model was not a suitable choice as it was too slow to train and exhibited a worse classification performance than the Naıve Bayes model on the Twitter datasets. The results of my investigation during the internship then led on to this project, where I aim to compare several different semi-supervised learning algorithms to enhance the performance of the Multinomial Naıve Bayes model implemented in DUALIST.

I implemented 3 different semi-supervised algorithms, Expectation-Maximization, Semi-supervised Frequency Estimate and Feature Marginals, together with a Multinomial Naıve Bayes classifier and investigated their performance on 16 real-world text corpora. 15 of these datasets contain Twitter data and were used to test the implemented algorithms in their target domain. The remaining dataset, the Reuters ApteMod corpus, was used for validating my implementations.

In addition to the comparison of the algorithms, I further investigated a variety of techniques to improve the performance of the classifier itself and the active learning process as a whole. These methods include the usage of uniform class priors for classification, randomly undersampling the majority class in case of class imbalances, optimising the free parameters in the Expectation-Maximization algorithm and using different uncertainty sampling techniques in the active learning process.

As the results show, using a Multinomial Naıve Bayes classifier, together with the Feature Marginals algorithm, generally produced the best results across datasets and configurations. Performance improvements could also be achieved by undersampling the majority class and by using an alternative uncertainty sampling technique in the active learning process. Using uniform class priors did not improve performance, and optimising the free parameters in the Expectation-Maximization algorithm is crucial for its classification performance.

<a href="{static}/pdfs/papers/bsc_diss.pdf" target="_blank" class="label label-primary">bsc dissertation</a> <a href="{static}/bibtex/Kober_2014.bib" target="_blank" class="label label-info">bibtex</a>