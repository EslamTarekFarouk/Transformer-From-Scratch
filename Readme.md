# Transformer from scratch
<h2>project description :</h2>
<br>
<p style = "font-family:Cursive">The goal of this project is to develop a comprehensive solution for classifying toxic comments using a Transformer model implemented from scratch using TensorFlow. The project focuses on addressing the problem of identifying and categorizing toxic comments on wikipedia talk pages.
The <a href = "https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data" style = "color:red"> dataset</a> used for this project contains comment samples and their corresponding labels. Each comment has a text associated with it, and there are six additional binary columns representing different types of toxic behavior: toxic, severe_toxic, obscene, threat, insult, and identity_hate.
To achieve the objective of classifying toxic comments, the project follows a structured approach where the code is organized into functions. This enhances modularity, readability, and maintainability throughout the project. The functions are designed to handle various tasks, such as data preprocessing, model architecture creation, training, and evaluation.</p>
<br>
<h2>Embeddings</h2>
<p style = "font-family:Cursive">To build a transformer model, we need to use pre-trained embeddings. This is because transformers are able to learn long-range dependencies in text, but they need some help understanding the meaning of individual words.both our data and the embeddings was collected from wikipedia.you can download the embeddings from  here <a href = "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.vec" style = "color:red">Click me
</a>
</p>
<p style = "font-family:Cursive">Before moving to the model architecture, I want to show why exactly I have used fast text with simple Wikipedia English words as embedding for my model.
the most important reason is that both contain words from Wikipedia which means we will not face problems like out-of-vocabulary much.
<br>
i have test the words that are out-of-vocabulary after cleaning the data and it turns out to be 6.73% which is very small . but, how can we deal with that? the fast embeddings came with a token for out-of-vocabulary words "/s token" It works like the mean vector of all vectors 
</p>
<h2>Scentences Variable length problem</h2>
<h4>Proplem</h4>
<p style = "font-family:Cursive">The problem of variable sentence length arises when using transformer models because transformers expect a fixed input length. This is because transformers use a self-attention mechanism, which allows each token in the input sequence to attend to every other token in the sequence. This requires the transformer to know the length of the input sequence in order to compute the attention weights.
</p>
<h4>Solution</h4>
<p style = "font-family:Cursive">One way to solve the problem of variable sentence length is to use padding. Padding is the process of adding special tokens to the beginning or end of a sentence to make it a fixed length. For example, if we set the maximum sentence length to 50 tokens, we would pad any sentences that are less than 50 tokens long with special padding tokens.</p>
<h4>Proplem</h4>
<p style = "font-family:Cursive">
it turns out that 25% of the scentences have less that 9 words , 50% less than 18 and 75% less than 38.we have two options now either to take the number of words to be a large number which might lead to a lot of paddings or to take the number of words to be small which might lead to truncating long sentences and hence lose their meaning.
</p>
<h4>Solution</h4>
<p style = "font-family:Cursive">fortunately, we can deal with the problem of padding by something called mask padding which will be covered later.so, our decision would be the value that is larger than 75% of all the lengths something like 50</p>

<h2>Model Architecture </h2>
<p style = "font-family:Cursive">All the ideas and architectural design is based on 4 research papers <a href = "https://arxiv.org/abs/2304.10557#:~:text=An%20Introduction%20to%20Transformers%20Richard%20E.%20Turner%20The,natural%20language%20processing%2C%20computer%20vision%2C%20and%20spatio-temporal%20modelling.">An Introduction to Transformers</a> , <a href = "https://arxiv.org/abs/2106.04554#:~:text=A%20Survey%20of%20Transformers%20Tianyang%20Lin%2C%20Yuxin%20Wang%2C,lots%20of%20interest%20from%20academic%20and%20industry%20researchers.">A Survey of Transformers</a> , <a href = "https://arxiv.org/abs/2305.05627#:~:text=An%20Exploration%20of%20Encoder-Decoder%20Approaches%20to%20Multi-Label%20Classification,have%20proven%20more%20effective%20in%20other%20classification%20tasks.">An Exploration of Encoder-Decoder Approaches to
Multi-Label Classification</a> and <a href = "https://arxiv.org/abs/1706.03762">Attention Is All You Need</a>.Generally, the Transformer architecture can be used in three different ways:
</p>   
<br>
<ul style = "font-family:Cursive">
<li>Encoder-Decoder: The full Transformer architecture. This is
typically used in sequence-to-sequence modeling (e.g., neural machine translation).
</li>
<li>Encoder only: Only the encoder is used and the outputs of the encoder are utilized as a
representation for the input sequence. This is usually used for classification or sequence
labeling problems.</li>
<li>Decoder only: Only the decoder is used, where the encoder-decoder cross-attention module is
also removed. This is typically used for sequence generation, such as language modeling.
</li>
</ul>
<p style = "font-family:Cursive">The best architecture fit the task which is a multilabel classification problem  is Encoder-only-architecture  which would consist of the following components :
<br>
1- Stack of Encoders "m layers encoder"
<br>
2- Head "multi-label-classifier NN" 
</p>
<h3>Positional Encoding</h3>

$$\mathbf{PE}(position,i) \ = \ sin(\frac{position}{c^{\frac{i}{d}}}) \mathit{;i \ \ even}$$ 

$$\mathbf{PE}(position,i) \ = \ cos(\frac{position}{c^{\frac{i}{d}}}) \mathit{;i \ \ odd}$$
 

<p style = "font-family:Cursive">
d is the dimension of embeddings, the position represents the word position in the sentence, and at each word position we add a new positional embedding vector where each value of the vector values depends on both the position of the word and the dimension index in addition to a predefined parameter called c, we choose to put it 10000 which is the same value that was introduced in  Attention Is All You Need a paper
</p>
<h3>Full Process</h3>
<p style = "font-family:Cursive">The following process happens inside the transformer block m times such that each layer output is the input to the next layer</p>
    
$$Layer Norm(X^{(m-1)}) = \bar{X}^{(m-1)}$$

$$Y^{m} = X^{(m-1)} + MHSA_{\theta }( \bar{X}^{(m-1)} )$$ 

$$Layer Norm(Y^{(m)}) = \bar{Y}^{(m)}$$

$$X^{(m)} = Y^{m} + FeedForwardNN ( \bar{Y}^{(m)} )$$

$$Y_{hat} = Sigmoid(W_{c1} \ X^{(M)} +b_{c1})$$
    
<p style = "font-family:Cursive">We need to apply Dropout the same way the paper "Attention is all you need" suggested :
    <br>
Residual Dropout We apply dropout [27] to the output of each sub-layer, before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of
P_drop = 0.1.
 <br>
the reason for doing this is to avoid model overfitting
</p>
    
<h3>Multi Head Self Attention</h3>

$$X^{(m)} = Transformer \ block(X^{(m-1)}) \ ; \ where \ m \ is \ number \ of \  layes$$

$$X_{d , n}^{(m)} = \begin{pmatrix}
x_{11} & x_{12} & x_{1n} \\
x_{21} & x_{22} & x_{2n} \\
. & . & . \\
x_{d1} & x_{d2} & x_{dn} \\
\end{pmatrix}$$

$$MHSA_{\theta}(X^{(m-1)}) = Concat_{h = 1}^{H}(V_{h} \  X^{(m-1)} \ A_{h}^{(m)}) \ W$$

$$A_{h}^{(m)} = \mathit{SoftMax}(\frac{(Q_{h}^{m})^{T} \ K_{h}^{m} + M^{m}}{\sqrt{k}})$$

$$Q_{h}^{m} = U_{q,h}^{m} \  X^{(m-1)}\\
K_{h}^{m} = U_{k,h}^{m} \; X^{(m-1)}
$$

<p style = "font-family:Cursive">Notice that m represents the layer number such that each layer output is the input to the next layer, you can think of it as a stack of layers.<br>
h is the number of heads, we can use multiple heads to capture more information "attention weights<br>
Q is the Query matrix and K is the key matrix both of size k x n<br>
we must apply the softmax function on the attention matrix to get attention capabilities across each column
 </p>
 
 $$\sum_{\acute{n}= 1}^{N} A_{\acute{n},n} = 1$$

<p style = "font-family:Cursive">All of these matrixes are parameters and we are going to make the model learn them through backpropagation. so, our parameter space of the MHSA would be as follows :</p>

$$\theta  = \begin{Bmatrix} U_{q,h},U_{k,h}, V_{h}
\end{Bmatrix}_{h = 1}^{H}$$

<p style = "font-family:Cursive">another ambiguous part is the M matrix, what is the purpose of the M matrix?
M matrix is used to mask the attention matrix, do you remember how we could deal with the variable length of the sentences we had to choose some fixed length of the sentence and we choosed the second approach which uses a long representation and we have claimed this approach will have a problem of adding a lot of paddings to the short sentences, and I promised to solve this problem and here we are using the mask which is simply a matrix of size N x N where each column that has padding in it is filled with ones and other values with zeros then we take that mask and multiply it with very large negative number say -10^9 and add that to the resultant matrix of the attention, after applying the softmax function all the columns that represent padding are given equal attention that is divided by their number which would be very close to zero as N increase  </p>
<p style = "font-family:Cursive">let say we have the following input embedding : </p>

$$X = \begin{pmatrix}
good &night & <PAD> \\
0.3 & 0.4 & 0 \\
0.2 & 0.5 &  0\\
0.9 &  0.8&  0\\
. &  .&  .\\
\end{pmatrix}$$

<p style = "font-family:Cursive">it's corresponding mask will be as follows : </p>

$$M_{x} = \begin{pmatrix}
good &night & <PAD> \\
0 & 0 & 1 \\
0 & 0 &  1\\
0 &  0&  1\\
. &  .&  .\\
\end{pmatrix} * -10^{9}$$

<h3>Layer Normalization</h3>

$$Layer Norm(X)_{d,n} = \frac{x_{d,n}-mean(x_{d})}{\sqrt{Var(X_{d})}}$$

<p style = "font-family:Cursive">Normalize the output of multihead self attention then add it to the input X</p>

<h3>Feed forward NN</h3>
<p style = "font-family:Cursive">this NN has 1 hidden layer with dimension d and Dropout layer</p>

$$\left [ Z_{1}= Relu(W_{1} X + b_{1}) \right ]\rightarrow \left [ Z_{2}= Dropout(Z_{1}) \right ]\rightarrow \left [ Y = W_{2}Z_{2} + b_{2} \right ]$$