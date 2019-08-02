Title: About me
save_as: index.html
status: hidden

I am currently post-doccing at the University of Edinburgh and I have previously PhDed with focus on Compositional Distributional Semantics at the University of Sussex.

<center>
<img src="{static}/images/ProfilePicture_small.png" alt="drawing" height="182" width="160"/>
</center>

See a recent CV <a href="{static}/pdfs/cv.pdf" target="_blank">here</a> (_last updated 23rd July 2019_) or feel free to [contact](./contact) me if you have any queries about my work or want to talk about NLP in general.

I am currently also co-organising the <a href="https://www.meetup.com/PyData-Edinburgh/" target="_blank">PyData Edinburgh Meetup</a>. If you are interested in giving a main talk or a ligthning talk about something related to data and/or python, please [get in touch](./contact). 

## About my Research

My research is focused on building systems that understand natural language. I am particularly interested in modelling _causation and consequence_. Identifying a change of state for some entities and inferring their consequent state is a pretty tricky problem, that involves <a href="https://en.wikipedia.org/wiki/Tense–aspect–mood" target="_blank">tense, aspect, mood</a>, and all sorts of temporality in general.

Modelling these fine-grained semantic properties at the word, clause and sentence level is very difficult and pretty much all current models fail to achieve reasonable performance on even a simple dataset (see our recent paper [here]({static}/pdfs/papers/TemporalAndAspectualEntailment_IWCS.pdf)).

For example, it sounds reasonable at first to assume that _visits_ entails _be at/in_, so whenever we encounter the predicate _visit_ we infer that whoever is doing the visiting must also _be_ at that place.

However, when temporality comes into play, things get a bit more complicated, as _Elizabeth will arrive at Pemberley_ does not entail that she _is at_ Pemberley _now_.

Things are also uncertain when we read that _Elizabeth had arrived at Pemberley_. Its unclear whether she's still there, so all we can infer is that _Elizabeth WAS at Pemberley_ at some point in the past as we could easily add _but she left 3 days later_.

Only when we read that _Elizabeth has arrived at Pemberley_ can we infer that _Elizabeth IS at Pemberley_ at the time of utterance. Thus the _consequent state_ of having arrived at a location is _being_ at that location (now).

And thats just the beginning! Another class of difficult unidirectional inferences concern the <a href="https://en.wikipedia.org/wiki/Lexical_aspect" target="_blank">Aktionsart</a> of a verb. Often entailment between verbs follows paraphrastic patterns, such as _buying a thing_ entails _purchasing that thing_ or _acquiring that thing_, and _vice versa_ in all combinations. All of these verbs, _buy_, _purchase_ and _acquire_, are _actions_, but when we throw in a _state_, the bidirectionality of the entailment relations breaks down. 

For example, while _Mr Bingley buys Netherfield_ entails _he purchases Netherfield_ and _vice versa_, the entailment between _Mr Bingley buys Netherfield_ and _Mr Bingley owns Netherfield_ is one way only, i.e. _Mr Bingley owns Netherfield_ does _NOT_ entail _he bought it_, as he could have also _built it_ himself, or _inhereted_ it. Ownership is another example of a _consequence_ or _consequent state_ of an event - the purchase of Netherfield. 

Being able to model temporality is useful for tracking the state of the world (i.e. building and maintaining the state of a knowledge base), as it enables the correct inference of facts about ownership or by following the whereabouts of certain entities in text.

For example, temporality is important to draw the correct inferences from the two following (semi-fictional) news headlines, _Google has acquired YouTube_ and _Amazon will buy Netflix_.

Inferring that in the former case, the acquisition of YouTube is completed, and the consequences of the acquisition (ownership of YouTube by Google), hold at the time of utterance, it is possible to answer questions such as _Who owns YouTube?_. In the latter case, we would be able to infer that Amazon does not (yet) own Netflix, hence our model would be able to correctly answer the question _Does Amazon own Netflix?_ with _no_.

Another interesting problem is tracking _where_ a particular person currently is, for example given the following (fictional) snippet of news text:

_The first minister Nicola Sturgeon has arrived in Edinburgh late last night. However, she will travel onwards to Brussels early on Tuesday in order to meet the President of the European Commission Ursula von der Leyen, followed by a brief visit to Berlin to meet with chancellor Angela Merkel on Wednesday._

Modelling temporality is essential for answering the question _Where is Nicola Sturgeon now?_ with _Edinburgh_.
