# Lecture 16: Text Representations — From Words to Meaning

> **Worksheet File**: [Open Lecture 16 Interactive Worksheet](file:///Users/yashagarwal/Downloads/Newton%20College/Deep%20Learning/Lecture/Lecture_16_Text_Representations/Lecture_16_Text_Representations.html)  
> **Course Progression**: Lecture 15 (Gated RNNs: LSTM/GRU) &rarr; **Lecture 16 (Text Representations & Embeddings)** &rarr; Lecture 17 (Encoder-Decoder & Attention)

---

## 1. The Basics: Two Opening Puzzles

Before feeding text into any neural network, we must convert raw text strings into numerical vectors. But how we represent words determines what information is preserved and what is lost.

### Puzzle 1: Order Collapse
Consider two sentences:
* Sentence A: `"dog bites person"`
* Sentence B: `"person bites dog"`

If we count words:
* `dog`: 1, `bites`: 1, `person`: 1 in both sentences.
* The words match, but the event is completely different. A simple count vector erases **token order** (who appears before or after whom).

### Puzzle 2: The Semantic Gap
Consider two words:
* `"cat"` and `"kitten"`
* The character strings are completely distinct, but their underlying meaning strongly overlaps. A naive representation treats them as entirely unrelated.

```
       Input Text
           │
           ▼
     [Tokenization]        ──▶ Split string into discrete units & map to IDs
           │
           ▼
    [Representation]
     ├── Discrete / Count  ──▶ BoW, TF-IDF (Sparse, order-blind, vocabulary-sized)
     └── Continuous Vector ──▶ Embeddings (Dense, semantic coordinates, dimension d)
```

---

## 2. Core Vocabulary & Foundations

* **Corpus**: A collection of documents.
* **Token**: One chosen text unit (a character, a subword, or a word).
* **Vocabulary ($V$)**: The fixed set of known tokens mapped to integer IDs: $\{0, 1, \dots, |V| - 1\}$.
* **Vector**: An ordered list of real numbers; a **matrix** is a 2D table of numbers.
* **Sequence**: An ordered list of tokens where position matters.
* **Timestep**: Processing one sequential token (does not imply wall-clock time).

---

## 3. Tokenization: Where Does a Word Begin?

Tokenization maps raw text $x$ into an ordered sequence of tokens, then translates each token string to an integer ID via a vocabulary dictionary:

$$\text{Tokenize}(x) = (t_1, t_2, \dots, t_L)$$
$$\text{id}_k = \text{vocabulary}[t_k]$$

Where:
* $L$ is the sequence length.
* $t_k$ is the $k$-th token string.
* $\text{id}_k$ is the integer index in the vocabulary table.

### Tokenization Approaches & Trade-Offs

| Level | Example (`"unhappiness!"`) | Advantage | Disadvantage |
| :--- | :--- | :--- | :--- |
| **Word Tokens** | `["unhappiness", "!"]` | Intuitive, keeps semantic units intact | Huge vocabulary ($|V| > 100\text{k}$); cannot handle unknown/unseen words (OOV) |
| **Character Tokens** | `["u", "n", "h", "a", "p", ...]` | Tiny vocabulary; can spell any unseen word | Sequences become very long ($L \times 5$ to $10$), straining memory |
| **Subword Tokens (BPE)** | `["un", "happiness", "!"]` | Balances vocabulary size and sequence length; frequent words stay intact, rare words split into pieces | Requires training a merge table on corpus |

### Miniature Byte-Pair Encoding (BPE) Example
How subwords are learned from corpus frequencies:
* Training corpus: three words `low`, `low`, `lower`, each ending in an end-of-word marker `▁`:
  * `l o w ▁` (count = 1)
  * `l o w ▁` (count = 1)
  * `l o w e r ▁` (count = 1)
* **Merge 1**: Count all adjacent symbol pairs. The pair `(l, o)` occurs 3 times. Merge `l + o` $\to$ `lo`.
* **Merge 2**: Count pairs again. The pair `(lo, w)` occurs 3 times. Merge `lo + w` $\to$ `low`.
* **Merge 3**: The pair `(low, ▁)` occurs 2 times. Merge `low + ▁` $\to$ `low▁`.
* **Result**: We now have a dedicated token `low▁` for the frequent complete word, and a reusable subword `low` that helps represent `lower▁` as `["low", "er▁"]`.

### Essential Exam Checks
* **Word & Punctuation Tokenization**: Tokenizing `"Cats, café!"` with lowercasing yields **4 tokens**: `cats`, `,`, `café`, `!`. Lowercasing standardizes casing, but punctuation symbols are preserved as individual tokens.
* **Integer IDs are Shelf Addresses**: If the vocabulary assigns `cat` $\to \text{id } 8$ and `dog` $\to \text{id } 9$, the numerical difference $9 - 8 = 1$ is an arbitrary dictionary index. **Numeric IDs do not encode semantic distance or similarity.**

---

## 4. Bag-of-Words (BoW): Counting Without Order

The Bag-of-Words representation constructs a $|V|$-dimensional vector $x \in \mathbb{R}^{|V|}$ where each coordinate $x_j$ counts how many times vocabulary word $v_j$ occurs in the document:

$$x_j = \sum_{k=1}^L \mathbb{I}[t_k = v_j]$$

Where $\mathbb{I}[\cdot]$ is the indicator function ($1$ if true, $0$ if false).

### Worked Numerical Examples
Given vocabulary: `[dog, bites, person]` ($|V| = 3$):
1. `"dog bites person"`:
   $$x = [1, 1, 1]^T$$
2. `"person bites dog"`:
   $$x = [1, 1, 1]^T$$
   *Both reversed sentences produce the exact same count vector!* Unigram BoW is completely blind to word order.
3. `"dog dog bites"`:
   $$x = [2, 1, 0]^T$$
   *The `dog` coordinate is 2 because the indicator fires twice.*

### Limitations of BoW
* **High Dimensionality & Extreme Sparsity**: With a real-world vocabulary ($|V| \ge 20,000$), a short 10-word document is a vector of 20,000 numbers with 19,990 zeros.
* **Stopword Domination**: Common grammatical words (`the`, `is`, `and`) have huge counts, overshadowing informative content words.

---

## 5. TF-IDF: Downweighting Ubiquitous Words

Term Frequency-Inverse Document Frequency (TF-IDF) adjusts raw word counts so that words appearing in almost every document receive low weights, while words concentrated in specific documents receive high weights.

### The Mathematical Formulas

1. **Term Frequency (TF)**: Raw count of term $t$ in document $d$:
   $$\text{TF}(t, d) = \text{count}(t, d)$$

2. **Smoothed Inverse Document Frequency (IDF)**:
   $$\text{IDF}(t) = \ln\left(\frac{N + 1}{\text{df}(t) + 1}\right) + 1$$
   * $N$: Total number of documents in the corpus.
   * $\text{df}(t)$: Document frequency (how many documents contain term $t$ at least once).
   * Adding $+1$ inside the logarithm prevents division by zero (smoothing).
   * Adding $+1$ at the end ensures terms present in all documents still receive a non-zero weight of $1.0$.

3. **Composite TF-IDF Score**:
   $$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

### Complete Worked Corpus Example
Given a corpus of $N = 3$ documents:
* $D_1$: `"the cat likes milk"`
* $D_2$: `"the dog likes walks"`
* $D_3$: `"the cat likes walks"`

Let's compute the IDF for every term:
* `the`: Appears in $D_1, D_2, D_3 \implies \text{df} = 3$:
  $$\text{IDF}(\text{the}) = \ln\left(\frac{3 + 1}{3 + 1}\right) + 1 = \ln(1) + 1 = 0 + 1 = 1.000$$
* `likes`: Appears in $D_1, D_2, D_3 \implies \text{df} = 3$:
  $$\text{IDF}(\text{likes}) = \ln(1) + 1 = 1.000$$
* `cat`: Appears in $D_1, D_3 \implies \text{df} = 2$:
  $$\text{IDF}(\text{cat}) = \ln\left(\frac{3 + 1}{2 + 1}\right) + 1 = \ln\left(\frac{4}{3}\right) + 1 \approx 0.2877 + 1 = 1.288$$
* `walks`: Appears in $D_2, D_3 \implies \text{df} = 2$:
  $$\text{IDF}(\text{walks}) = \ln\left(\frac{4}{3}\right) + 1 \approx 1.288$$
* `dog`: Appears only in $D_2 \implies \text{df} = 1$:
  $$\text{IDF}(\text{dog}) = \ln\left(\frac{3 + 1}{1 + 1}\right) + 1 = \ln(2) + 1 \approx 0.6931 + 1 = 1.693$$
* `milk`: Appears only in $D_1 \implies \text{df} = 1$:
  $$\text{IDF}(\text{milk}) = \ln(2) + 1 \approx 1.693$$

### Key Exam Questions on TF-IDF Behavior
1. **Repeating a word within the same document**: If we repeat `"milk"` 5 times in $D_1$ without adding it to any other document, what changes?
   * *Answer*: Only its **TF** in $D_1$ increases (from 1 to 5). Its document frequency ($\text{df}$) remains 1 because $\text{df}$ counts *documents*, not occurrences.
2. **Adding a word to a new document**: If `"milk"` is added to $D_2$, what changes?
   * *Answer*: Its $\text{df}$ increases from $1 \to 2$, which reduces its IDF from $1.693 \to 1.288$ for *all* documents in the corpus.

---

## 6. Dense Embeddings & Vector Space Geometry

Instead of high-dimensional sparse count vectors ($\mathbb{R}^{|V|}$), word embeddings assign each token a low-dimensional, dense, continuous vector in $\mathbb{R}^d$ (typically $d \in [50, 300]$).

### The Embedding Matrix ($E$)
* Let vocabulary size be $|V|$ and embedding dimension be $d$.
* The embedding table is a matrix:
  $$E \in \mathbb{R}^{|V| \times d}$$
* Looking up a token $\text{id}_k$ selects row $k$:
  $$e_k = E[\text{id}_k] \in \mathbb{R}^d$$
* **One-Hot Equivalence**: Looking up row $k$ is mathematically identical to multiplying a $1 \times |V|$ one-hot vector $o_k$ by matrix $E$:
  $$e_k = o_k E$$
* **Parameter Count**: For $|V| = 10,000$ and $d = 128$, matrix $E$ contains:
  $$10,000 \times 128 = 1,280,000 \text{ trainable parameters}$$

### Cosine Similarity: Measuring Angular Alignment
The semantic similarity between two dense vectors $a, b \in \mathbb{R}^d$ is measured by the cosine of the angle between them:

$$\cos(a, b) = \frac{a \cdot b}{\|a\|_2 \|b\|_2} = \frac{\sum_{i=1}^d a_i b_i}{\sqrt{\sum_{i=1}^d a_i^2} \sqrt{\sum_{i=1}^d b_i^2}}$$

* $\cos(a, b) = 1.0$: Pointing in identical directions ($0^\circ$).
* $\cos(a, b) = 0.0$: Orthogonal / unrelated ($90^\circ$).
* $\cos(a, b) = -1.0$: Pointing in opposite directions ($180^\circ$).
* **Why Cosine?** It normalizes by vector length $\|a\|_2$, making the measure invariant to word frequency or norm scaling.

### Static Embedding Limitation (Polysemy)
A static lookup table $E$ assigns a **single fixed vector** to each token string. In the sentences:
1. `"I deposited money in the bank"`
2. `"We sat on the river bank"`

A static embedding model assigns the exact same vector $E[\text{bank}]$ to both occurrences, unable to distinguish financial institution from river edge without a context-sensitive sequence model.

---

## 7. Word2Vec: Continuous Bag of Words (CBOW)

The foundational insight of Word2Vec (Mikolov et al., 2013) is the **Distributional Hypothesis**: *"You shall know a word by the company it keeps"* (Firth, 1957). Words appearing in similar contexts should have similar vector representations.

### CBOW Architecture
CBOW predicts a missing **center target word** from the surrounding **context words**.

```
Context Words: [ the,  milk ]
                      │
           (Lookup in E and Average)
                      │
                      ▼
               h = (1/C) Σ e_c       (Average context state in R^d)
                      │
             (Project: z = U h + b)
                      │
                      ▼
               z ∈ R^|V|             (Raw logit score per vocabulary word)
                      │
                   (Softmax)
                      │
                      ▼
               p = softmax(z)        (Probability distribution over |V|)
                      │
              Loss: -ln p[target]
```

### Mathematical Formulation
1. **Context Aggregation**: Given context tokens $\{c_1, \dots, c_C\}$, compute the mean vector $h \in \mathbb{R}^d$:
   $$h = \frac{1}{C} \sum_{c \in \text{context}} E[c]$$
2. **Score Computation**: Project $h$ to vocabulary size $|V|$ using output matrix $U \in \mathbb{R}^{|V| \times d}$ and bias $b \in \mathbb{R}^{|V|}$:
   $$z = U h + b$$
3. **Probability Distribution**:
   $$p = \text{softmax}(z), \quad p_i = \frac{\exp(z_i)}{\sum_{j=1}^{|V|} \exp(z_j)}$$
4. **Cross-Entropy Training Objective**:
   $$\mathcal{L} = -\ln p[\text{target}]$$
5. **Parameter Updates**:
   $$\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$$

### Complete Worked Forward Pass (From HTML Worksheet)
Let vocabulary be `[the, milk, cat]` ($|V| = 3, d = 2$):
* Input embedding rows in $E$:
  * `the`: $[1.0, 0.0]$
  * `milk`: $[0.0, 1.0]$
  * `cat`: $[0.2, 0.4]$
* Target word: `cat` (index 3). Context words: `[the, milk]` ($C = 2$).

**Step 1: Compute average context vector $h$**:
$$h = \frac{1}{2} ([1.0, 0.0]^T + [0.0, 1.0]^T) = [0.5, 0.5]^T$$

**Step 2: Compute logits $z = U h + b$** (with output weights $U$ and zero biases $b = [0, 0, 0]^T$):
* $U_{\text{the}} = [0.2, -0.1] \implies z_1 = (0.2)(0.5) + (-0.1)(0.5) = 0.10 - 0.05 = 0.05$
* $U_{\text{milk}} = [-0.3, 0.4] \implies z_2 = (-0.3)(0.5) + (0.4)(0.5) = -0.15 + 0.20 = 0.05$
* $U_{\text{cat}} = [0.1, 0.2] \implies z_3 = (0.1)(0.5) + (0.2)(0.5) = 0.05 + 0.10 = 0.15$
$$z = [0.05, 0.05, 0.15]^T$$

**Step 3: Softmax probabilities**:
* $\exp(0.05) \approx 1.0513$
* $\exp(0.05) \approx 1.0513$
* $\exp(0.15) \approx 1.1618$
* Sum $= 1.0513 + 1.0513 + 1.1618 = 3.2644$
* $p_1 = \frac{1.0513}{3.2644} \approx 0.322$
* $p_2 = \frac{1.0513}{3.2644} \approx 0.322$
* $p_3 = \frac{1.1618}{3.2644} \approx 0.356$
$$p = [0.322, 0.322, 0.356]^T$$

**Step 4: Target Loss (for `cat`)**:
$$\mathcal{L} = -\ln(p_3) = -\ln(0.3559) \approx 1.033$$

**Step 5: Gradient Flow**:
* Output gradient: $g = p - y_{\text{onehot}} = [0.322, 0.322, 0.356 - 1.0]^T = [0.322, 0.322, -0.644]^T$
* Gradients on parameters: $\frac{\partial \mathcal{L}}{\partial U} = g h^T$, $\frac{\partial \mathcal{L}}{\partial b} = g$, $\frac{\partial \mathcal{L}}{\partial h} = U^T g$
* Gradient distributed back to input embeddings: each context token receives $\frac{1}{C} \frac{\partial \mathcal{L}}{\partial h}$.

*(Note: Skip-Gram reverses the prediction task: it takes the center token as input and predicts each context token).*

---

## 8. GloVe: Global Vectors from Co-occurrence Statistics

While CBOW trains by streaming local windows, **GloVe** (Pennington et al., 2014) builds a global co-occurrence matrix $X$ from the whole corpus and optimizes directly on it.

### Step 1: Building the Co-occurrence Matrix ($X$)
Using a symmetric window of radius 1:
* In the string `"cat dog cat"`:
  * For position 1 (`cat`): right neighbor is `dog` $\implies X_{\text{cat}, \text{dog}} += 1$
  * For position 2 (`dog`): left neighbor is `cat`, right neighbor is `cat` $\implies X_{\text{dog}, \text{cat}} += 2$
  * For position 3 (`cat`): left neighbor is `dog` $\implies X_{\text{cat}, \text{dog}} += 1$
* Result: $X_{\text{cat}, \text{dog}} = 2$ and $X_{\text{dog}, \text{cat}} = 2$.
* **Rule**: Never cross document boundaries when counting co-occurrences.

### Step 2: The GloVe Objective Function

$$J = \sum_{X_{ij} > 0} f(X_{ij}) \left( w_i \cdot \tilde{w}_j + b_i + \tilde{b}_j - \ln X_{ij} \right)^2$$

Where:
* $w_i \in \mathbb{R}^d, b_i \in \mathbb{R}$: Word vector and bias for target word $i$.
* $\tilde{w}_j \in \mathbb{R}^d, \tilde{b}_j \in \mathbb{R}$: Context vector and bias for context word $j$.
* $X_{ij}$: Co-occurrence count of word $i$ with word $j$.
* **Why omit $X_{ij} = 0$?** Because $\ln(0)$ is mathematically undefined ($-\infty$). The summation strictly iterates over non-zero pairs ($X_{ij} > 0$).
* **Weighting Function $f(x)$**:
  $$f(x) = \min\left(1, \left(\frac{x}{x_{\max}}\right)^\alpha\right)$$
  (Typically $x_{\max} = 4 \text{ or } 100, \alpha = 0.75$). This prevents very frequent word pairs (e.g. `the` $\leftrightarrow$ `of`) from dominating the loss.

### Worked Example: Calculating One Objective Term (From HTML Worksheet)
Let:
* Word pair: `the` $\to$ `cat` with co-occurrence count $X_{ij} = 2$.
* Parameters: $w = [1, 0]^T$, $\tilde{w} = [0, 1]^T$, $b = 0.25$, $\tilde{b} = 0.25$.
* Hyperparameters: $x_{\max} = 4, \alpha = 0.75$.

1. **Predicted log-count**:
   $$\hat{y} = w \cdot \tilde{w} + b + \tilde{b} = (1)(0) + (0)(1) + 0.25 + 0.25 = 0.50$$
2. **Actual log-count**:
   $$\ln(X_{ij}) = \ln(2) \approx 0.69315$$
3. **Residual (error)**:
   $$\text{residual} = 0.50 - 0.69315 = -0.19315$$
4. **Weighting factor**:
   $$f(2) = \left(\frac{2}{4}\right)^{0.75} = (0.5)^{0.75} \approx 0.5946$$
5. **Loss Contribution**:
   $$J_{\text{pair}} = f(X_{ij}) \times (\text{residual})^2 = 0.5946 \times (-0.19315)^2 = 0.5946 \times 0.037307 \approx 0.0222$$

---

## 9. Downstream Applications & Evaluating Representations

### Document Retrieval via Mean Embedding
To represent an entire document $d$ of length $L$, a common baseline averages all token vectors:

$$\bar{d} = \frac{1}{L} \sum_{k=1}^L e_k$$

To rank documents against a user query $q$, compute cosine similarity:

$$\text{score}(q, d) = \cos(\bar{q}, \bar{d})$$

### The Fatal Vulnerability of Mean Embeddings
Because vector addition is commutative:
$$e_{\text{dog}} + e_{\text{bites}} + e_{\text{person}} = e_{\text{person}} + e_{\text{bites}} + e_{\text{dog}}$$
* The mean embedding $\bar{d}$ of `"dog bites person"` is **mathematically identical** to `"person bites dog"`.
* Similarly, `"good, not bad"` and `"bad, not good"` collapse to identical averages.
* **To preserve role sensitivity and word order, you must keep the ordered matrix of token vectors and pass it to a sequence model (RNN/LSTM/Transformer).**

### How to Evaluate Representations
* **Avoid**: Relying solely on visually pleasing 2D t-SNE / PCA projection plots.
* **Use**: Held-out relevance judgments (e.g. Recall@$k$, Mean Reciprocal Rank) and test targeted challenge slices (negation, rare domain terms, role reversals).

---

## 10. Summary Checklist & Exam Must-Knows

- [ ] **Tokenization**: Lowercasing `"Cats, café!"` produces 4 tokens: `cats`, `,`, `café`, `!`.
- [ ] **Token IDs**: Integer IDs are arbitrary lookup indices; $|id_A - id_B|$ has zero semantic meaning.
- [ ] **BoW Reversals**: `"dog bites person"` and `"person bites dog"` produce identical unigram count vectors $[1, 1, 1]^T$.
- [ ] **TF-IDF Smoothing**: $\text{IDF} = \ln((N+1)/(\text{df}+1)) + 1$. For $N=3, \text{df}=1 \implies \ln(2)+1 \approx 1.693$.
- [ ] **TF vs IDF Updates**: Repeating a word in one document changes only its TF; adding a word to a new document increases df and decreases IDF for all documents.
- [ ] **Embedding Table Size**: $E \in \mathbb{R}^{V \times d}$. For $|V|=10,000, d=128 \implies 1.28\text{M}$ parameters.
- [ ] **Static Lookup**: A static embedding assigns the exact same vector to a word regardless of whether it means a river bank or a financial bank.
- [ ] **CBOW Prediction**: Predicts center word from average context $h = \frac{1}{C}\sum e_c$. Loss is $-\ln p[\text{target}]$.
- [ ] **GloVe Objective**: $J = \sum_{X_{ij}>0} f(X_{ij})(w_i \cdot \tilde{w}_j + b_i + \tilde{b}_j - \ln X_{ij})^2$. Pairs with $X_{ij}=0$ are omitted because $\ln(0)$ is undefined.
- [ ] **Preserving Order**: Mean embeddings discard order; to distinguish `"customer refunded merchant"` from `"merchant refunded customer"`, use an ordered sequence of token vectors.
