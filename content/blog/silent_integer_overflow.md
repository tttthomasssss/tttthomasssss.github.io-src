Title: The Silence of the Integers
Date: 2019-04-12 21:15
Category: Blog
Tags: python, scipy, numpy, pandas, pytorch, tensorflow, machine learning, data science
Slug: silent-integer-overflow
Authors: Thomas
Summary: Integers overflow silently in a lot of popular python libraries (including numpy, scipy, pandas, pytorch, tensorflow).

While implementing my own <a href="https://github.com/tttthomasssss/wort/tree/dev_v0.2.0" target="_blank">count-based distributional semantics package</a> from scratch I noticed something peculiar when debugging the model: summing rows or columns in a scipy sparse matrix gives _really_ weird results!

I gave two lightning talks at the PyData Edinburgh Meetup regarding that issue (see [talks](../pages/talks) for slides), and I have two Jupyter Notebooks for reproducing all that work <a href="{static}/notebooks/silent_integer_overflow.ipynb" download>here</a> and <a href="{static}/notebooks/silence_of_the_integers.ipynb" download>here</a>.

For example, numpy is very good with type broadcasting, so if you do the following with a `np.uint8`, numpy will automagically upcast the datatype:

```python
import numpy as np
x = np.full((3,12), 255, dtype=np.uint8)
print(x)

# Output
# array([[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
#        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
#        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]], dtype=uint8) # <-- small dtype

print(x.sum(axis=1))

# Output
# array([3060, 3060, 3060], dtype=uint64) # <-- automatically upcasted dtype

```

Brilliant, it works in a pretty failsafe way (unless you are a bit short on memory) , if the chosen `dtype` is too small, numpy just upcasts to `uint64`. Fine, however back in 2016, scipy did not upcast the dtypes automatically, nor did it display any warnings or errors:

```python

from scipy import sparse
xs = sparse.csr_matrix(x) # <-- remember, thats the one with a dtype of np.uint8, with all elments of value 255

print(xs.sum(axis=1))

# Output 
# matrix([[244], [244], [244]], dtype=uint8) # <-- oopsy-daisy :/

```

Well, thats pretty bad as it can introduce nasty data bugs that are _really_ hard to catch. The workaround for that issue was upcasting the data manually before loading them into a sparse matrix:

```python

xx = sparse.csr_matrix((data.astype(np.uint16), (rows, cols)))
print(xx.sum(axis=1))

# Output
# matrix([[3060], [3060], [3060]], dtype=uint16)

```

But, nothing to worry, there was a <a href="https://github.com/scipy/scipy/issues/2534" target="_blank">Github Issue</a> reporting that when I came across the bug, and it has been fixed with scipy `0.18`. So nothing to worry about anymore.

**However**, I started poking around a little bit more and found some more worrying behaviour. Initially I expected that once the above issue is fixed, the other stuff I found would likely be resolved with the same fix, so I forgot about it for about 2 years until I tried to reproduce the issue out of curiosity. _Spoiler Alert:_ its still there.

So the following code still overflows silently in scipy:

```python

from scipy import sparse # tested with version 1.2.0
import numpy as np # tested with version 1.16.0

# Lets create some data
rows = np.array([1, 1, 1, 1, 2, 2, 3, 1, 1, 2], dtype=np.uint8)
cols = np.array([0, 1, 2, 3, 0, 2, 1, 0, 1, 2], dtype=np.uint8)
data = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 255], dtype=np.uint8)

# And a sparse matrix for these observations
S = sparse.csr_matrix((data, (rows, cols)))
print(S.A)

# Output
# array([[0, 0, 0, 0],
#        [2, 2, 1, 1],
#        [1, 0, 0, 0], # <-- huh? where is my data? (hint, it should be at 2/2)
#        [0, 1, 0, 0]], dtype=uint8)
```

Also, any arithmetic operation results in a silent overflow.

```python

# Lets create some more data...
rows = np.array([1, 1, 1, 1, 2, 2, 3, 1, 1, 2], dtype=np.uint8)
cols = np.array([0, 1, 2, 3, 0, 2, 1, 0, 1, 2], dtype=np.uint8)
data = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 13], dtype=np.uint8)

S = sparse.csr_matrix((data, (rows, cols)))
print(S.A)

# Output
# array([[ 0,  0,  0,  0],
#        [ 2,  2,  1,  1],
#        [ 1,  0, 14,  0],
#        [ 0,  1,  0,  0]], dtype=uint8)

# ...and some more data
rows = np.array([1, 2, 2, 2], dtype=np.uint8)
cols = np.array([0, 0, 2, 2], dtype=np.uint8)
data = np.array([1, 1, 1, 250], dtype=np.uint8)

T = sparse.csr_matrix((data, (rows, cols)), shape=S.shape)
print(T.A)

# Output
# array([[  0,   0,   0,   0],
#        [  1,   0,   0,   0],
#        [  1,   0, 251,   0],
#        [  0,   0,   0,   0]], dtype=uint8)

# Now lets do some adding and multiplying
U = S + T
print(U.A)

# Output
# array([[0, 0, 0, 0],
#        [3, 2, 1, 1],
#        [2, 0, 9, 0], # <-- hm, no, 251 + 14 is perhaps _not_ 9
#        [0, 1, 0, 0]], dtype=uint8) 
       
V = S * 20
print(V.A)

# array([[ 0,  0,  0,  0],
#        [40, 40, 20, 20],
#        [20,  0, 24,  0], # <-- while 14 * 20 is a lot, its perhaps _not_ 24
#        [ 0, 20,  0,  0]], dtype=uint8)

```

OK, so anything we do, if the `dtype` is too small, there will be a silent overflow in any scipy sparse matrix.

But surprisingly, its not just scipy that has a silent integer overflow, its also numpy!

```python

Ud = S.toarray() + T.toarray()
print(Ud)

# Output
# array([[0, 0, 0, 0],
#        [3, 2, 1, 1],
#        [2, 0, 9, 0], # <-- same as with scipy part 1
#        [0, 1, 0, 0]], dtype=uint8)
       

Vd = S.toarray() * 20
print(Vd)

# Output
# array([[ 0,  0,  0,  0],
#        [40, 40, 20, 20],
#        [20,  0, 24,  0], # <-- same as with scipy part 2
#        [ 0, 20,  0,  0]], dtype=uint8)

```

By that point I got slightly worried. While - arguably - integer overflow is going to be a rare edge case in any real world application, its _very_ worrying that its silent (and no, using `np.seterr('raise')`, does not solve the problem as it only works for scalars, but not arrays). Its possible that you load a chunk of data into numpy and because a few numbers were out of range you end up with nasty data bugs that are pretty much impossible to find.

But numpy and scipy surely must be outliers on that issue, other libraries will not suffer from silent integer overflow! Lets look at pandas:

```python

import pandas as pd # tested with version 0.23.1

x = np.array([2, 3, 4], dtype=np.uint8)
y = np.array([5, 6, 255], dtype=np.uint8)

df1 = pd.DataFrame(x)
df2 = pd.DataFrame(y)
df3 = df1.add(df2)

print(df3)

# Output
# 0	7
# 1	9
# 2	3 # <-- well no, the value in the last row is perhaps _not_ 3...

```

OK, one could argue that pandas has the same behaviour as numpy because its using a lot of numpy functionality under the hood. Surely stuff like pytorch will get it right!

```python

import torch # tested with version 1.0.1.post2

x = torch.randint(low=136, high=255, size=(1, 3), dtype=torch.uint8)
y = torch.randint(low=136, high=255, size=(1, 3), dtype=torch.uint8)
print(f'x={x}; y={y}')

# Output
# x=tensor([[214, 241, 245]], dtype=torch.uint8); y=tensor([[215, 167, 186]], dtype=torch.uint8)

x + y

# Output
# tensor([[173, 152, 175]], dtype=torch.uint8) <-- close, but not quite

# Somebody must notice something somewhere at some point, no?
z = torch.randint(low=666, high=999, size=(1, 3), dtype=torch.uint8) 
print(z)

# Output
# tensor([[156, 225, 155]], dtype=torch.uint8) <-- hmm, no
```

So, last resort TensorFlow, at this point I'm accepting bets on the outcome :).

```python
import tensorflow as tf # tested with version 1.12.0

x = tf.constant(value=np.array([1, 2, 255]), name='x', shape=(1, 3), dtype=tf.uint8)
y = tf.constant(value=np.array([3, 4, 2]), name='y', shape=(1, 3), dtype=tf.uint8)

session = tf.Session()
session.run(tf.global_variables_initializer())
print(session.run(tf.add(x, y)))

# Output
# [[4 6 1]] # <-- An honest effort but no

z = tf.constant(value=np.array([1000, 2000, 2550]), name='z', shape=(1, 3), dtype=tf.uint8)

print(session.run(z))

# Output
# [[232 208 246]] # <-- Again, A for effort, but F for output 

```

Now that we have established that everything fails, we could maybe ask _why_ it fails. A related and also interesting question is, why does it _NOT_ fail for floating point numbers? Well, the reason is floating point overflow is caught at hardware level. So the ALU sets a flag whenever floating point overflow is encountered and any client code can just check that flag. Thats _very_ efficient. The same thing does not exist for integers. So in numpy, basically _every single value_ would need to be checked after every operation that mutates the data. This is _very inefficient_, and thats why the numpy folks (and likely all others) don't check for it - its been known since at least 2008, see the <a href="https://github.com/numpy/numpy/issues/593">Github Issue</a> here. 

Anyways, its bad, and it can cause some hard to catch bugs, but overall its also a relatively rare edge case.