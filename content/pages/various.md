Title: Various
sortorder: 5

While updating this website I looked through some of my repositories and found a few cool gems. 

### TwoSips

I'm frequently converting images from one format to another (e.g. to add pictures of cats to my presentations) and my go-to solution has typically been to use <a href="https://ss64.com/osx/sips.html" target="_blank">`sips`</a> from the command line. However, I never remember its usage or options that I need to set and always have to look at the manual. Due to having been a Mac OS developer in a previous life, I got a bit cold-turkey on the absence of square brackets in python at some point and wrote a quick Objective-C App. Its still incomplete in terms of being polished, but its fully functional and I use it once every couple of months.

<a href="https://github.com/tttthomasssss/TwoSips" target="_blank" class="label label-success">code</a> <a href="{static}/other/TwoSips.zip" target="_blank" class="label label-danger">App</a> 

### Checkers

In my last undergrad year, one of our coursework assignments was to implement a Checkers game. 2-player games are a pretty cool AI problem and the way one implements tic-tac-toe or Checkers is essentially the same as one would produce a basic Chess engine. Once its figured out how a _Game state_ is represented and how _successors_ of that state are built, its all _MiniMax_ from then on. My checkers game implements <a href="https://en.wikipedia.org/wiki/Minimax" target="_blank">MiniMax</a> (actually <a href="https://en.wikipedia.org/wiki/Negamax" target="_blank">NegaMax</a>, because its slightly simpler implementationwise), together with <a href="https://en.wikipedia.org/wiki/Alpha–beta_pruning" target"_blank">Alpha-Beta Pruning</a>. The code is in Java, but I have also created a Python version - basically to check whether my answers on mock-exam papers are correct.

<a href="https://bitbucket.org/tttthomasssss/checkers/" target="_blank" class="label label-success">Checkers</a> <a href="https://github.com/tttthomasssss/GamePlayExample" target="_blank" class="label label-success">MiniMax Python</a>

### Braitenberg Vehicles

In my second undergrad year, we were playing around with _embodied AI_. One of the coolest things was implementing the <a href="https://en.wikipedia.org/wiki/Braitenberg_vehicle" target="_blank">Braitenberg Vehicles behaviour</a>, using Rodney Brooks' <a href="https://en.wikipedia.org/wiki/Subsumption_architecture" target="_blank">Submsumption Architecture</a> on Lego Mindstorm robots. The code is super simple and as part of my coursework submission I've filmed some of the Lego Mindstorm Braitenberg Vehicles in action.

<a href="https://bitbucket.org/tttthomasssss/lejos-mindstorms/" target="_blank" class="label label-success">Braitenberg Vehicles</a>

<iframe width="560" height="315" src="https://www.youtube.com/embed/PUrHW2jOtuI" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe> <iframe width="560" height="315" src="https://www.youtube.com/embed/y1LJbTLwvmg" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe> <iframe width="560" height="315" src="https://www.youtube.com/embed/36bCeG5Japo" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe> 