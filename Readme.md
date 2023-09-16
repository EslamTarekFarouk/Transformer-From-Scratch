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