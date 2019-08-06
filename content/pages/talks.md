Title: Talks
sortorder: 4

Slides and other resources for talks about individual papers can be found on the [Publications](./publications) page.

### 2019

I gave two lightning talks at our <a href="https://www.meetup.com/PyData-Edinburgh/" target="_blank">PyData Edinburgh Meetup</a> group on silent integer overflows. They are literally everywhere! I also wrote a short [blogpost](../silent-integer-overflow) on the subject!

<a href="{static}/pdfs/slides/PyDataEdinburgh_lightningtalk_Feb2019.pdf" target="_blank" class="label label-default">slides lightning talk feb</a> <a href="{static}/pdfs/slides/lightningtalk_april_pdf.pdf" target="_blank" class="label label-default">slides lightning talk apr</a>

### 2018

##### Tense and Aspect in Distributional Semantic Vector Space Models

_I talked about modelling tense and aspect in distributional semantic vector spaces at the University of Sussex in July. Unfortunately "The Aspect Paper" was rejected from EMNLP 2018 and is currently undergoing a major rehaul. However if you would like to read it, I can share it privately - [contact me](./contact) if you are interested._
<br /><br />
Tense and aspect are two of the most important contributing factors to the meaning of a verb, e.g. determining the temporal extent described by a predication as well as the entailments the verb gives rise to. For example while a verb phrase describing an event such as "Thomas is visiting Brighton" entails the change of state of "Thomas being in Brighton", the state of "being in Brighton" does not entail a "visit to Brighton". The reasoning becomes more complex when different tenses are involved, where "Thomas has arrived in Brighton" entails "Thomas is in Brighton", whereas "Thomas will arrive in Brighton" does not.

Distributional semantic word representations are an ubiquitous part for a number of NLP tasks such as Sentiment Analysis, Question Answering, Recognising Textual Entailment, Machine Translation, Parsing, etc. While their capacity for representing content words such as adjectives, nouns and verbs is well established, their ability to encode the semantics of closed class words has received considerably less attention.

In this talk I will show how composition can be used to leverage distributional representations for closed class words such as auxiliaries, prepositions and pronouns to model tense and aspect of verbs in context. I will furthermore analyse how and why closed class words are effective at disambiguating fine-grained distinctions in verb meaning. Lastly, I will demonstrate that a distributional semantic vector space model is able to capture a substantial amount of temporality in a novel tensed entailment task.

<a href="{static}/pdfs/slides/TenseAndAspectDSMs_Sussex.pdf" target="_blank" class="label label-default">slides</a>

<br /> 

##### Inferring Unobserved Co-occurrence Events in Anchored Packed Trees
_When I joined the ILCC at the University of Edinburgh at a post-doc I gave a talk about my PhD work on Anchored Packed Trees. They are a really exciting way of modelling composition in distributional vector spaces!_
<br /><br />
Anchored Packed Trees (APTs) are a novel approach to distributional semantics that takes distributional composition to be a process of lexeme contextualisation. An APT is represented with a higher-order dependency-typed structure (the APT) where paths associated with higher-order dependencies connect vertices associated with weighted lexeme multisets. The central innovation in the compositional theory is that the APT's type structure enables the precise alignment of the semantic representation of each of the lexemes being composed.

Like other count-based distributional spaces, APTs are prone to considerable data sparsity, caused by not observing all plausible co-occurrences in the given data. Most methods face this challenge in ways that render the resulting word representations uninterpretable, with the consequence that distributional composition becomes difficult to model and reason about.

In this talk, I will introduce Anchored Packed Trees and explain how distributional composition is modelled in the framework. An emperical evaluation of the framework on standard lexical and phrasal datasets highlights the extend to which data sparsity is impacting the performance of the model. In order to address the data sparsity challenge and retain the interpretability of the model, I explore an alternative algorithm - distributional inference - for improving elementary representations. I demonstrate how the algorithm can be embedded in the existing theory and empirically show that it improves elementary word representations and is especially beneficial when combined with an intersective composition function.

<a href="{static}/pdfs/slides/EdinburghNLP.pdf" target="_blank" class="label label-default">slides</a>

### 2017

##### _What does it all mean?_ - Compositional Distributional Semantics for Modelling Natural Language

_I spoke about compositional distributional semantics at the PyData Berlin conference. It was a fun exercise to talk to a non-NLP focused audience._

Representing words as vectors in a high-dimensional space has a long history in natural language processing. Recently, neural network based approaches such as word2vec and GloVe have gained a substantial amount of popularity and have become an ubiquituous part in many NLP pipelines for a variety tasks, ranging from sentiment analysis and text classification, to machine translation, recognising textual entailment or parsing.

An important research problem is how to best leverage these word representations to form longer units of text such as phrases and full sentences. Proposals range from simple pointwise vector operations, to approaches inspired by formal semantics, deep learning based approaches that learn composition as part of an end-to-end system, and more structured approaches such as anchored packed dependency trees.

In this talk I will introduce a variety of compositional distributional models and outline different approaches of how effective meaning representations beyond the word level can successfully be built. I will furthermore provide an overview of the advantages of using compositional distributional approaches, as well as their limitations. Lastly, I will discuss their merit for applications such as aspect oriented sentiment analysis and question answering.

<a href="{static}/pdfs/slides/pydata2017.pdf" target="_blank" class="label label-default">slides</a>

<br />

The talk has been recorded and published on <a href="https://www.youtube.com/channel/UCOjD18EJYcsBog4IozkF_7w" target="_blank">PyData's YouTube channel</a>.

<iframe width="560" height="315" src="https://www.youtube.com/embed/hTmKoHJw3Mg" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>