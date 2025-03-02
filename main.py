import json
import math
import string

from tqdm import tqdm
import numpy as np

# Tokens

def simple_tokenizer(data, *, token_size=255, forbidden_chars=string.punctuation):
    """
    Tokenizes input data into smaller chunks of specified size, removing specific characters.

    Args:
        data (str or list of str): The input text or list of texts to tokenize.
        token_size (int, optional): The maximum size of each token chunk. Default is 255.
        forbidden_chars (str, optional): Characters to be replaced with spaces. Default is punctuation.

    Returns:
        list: A list of tokenized chunks for each text input.

    1. Splits each token into chunks of size `token_size`.
    2. Converts text to lowercase.
    3. Replaces forbidden characters with spaces.
    4. Splits the text into words.
    5. Further splits each token into smaller chunks.
    6. Handles both single string and list inputs.
    """

    tokens = [
        [token[i:i + token_size] for token in
         text.lower().translate(str.maketrans(forbidden_chars, ' ' * len(forbidden_chars))).split()
         for i in range(0, len(token), token_size)]
        for text in (data if isinstance(data, list) else [data])
    ]

    return tokens

class EmbeddingData:
    """
    Class for generating word embeddings using co-occurrence matrix (COM), Point wise Mutual Information (PMI),
    and Singular Value Decomposition (SVD).

    Args:
        dataset (list of lists of str): The input dataset, where each list contains tokens (words).
        window_size (int): The size of the context window for co-occurrence calculation.
        pmi (bool): Whether to calculate Point wise Mutual Information (PMI) or not.
        smoothing (float): The smoothing value to avoid zero probabilities in PMI calculation.
        from_zero (bool): If True, sets negative PMI values to zero.
        d_model (int): The dimensionality of the final embedding space after SVD.

    1. Calculates co-occurrence matrix
    2. Calculates or not point-wise mutual information
    3. Calculates singular value decomposition
    """

    def __init__(self, *, embeddings = None, vocabulary = None):
        # Initializing empty placeholders for vocabulary and embeddings
        self.embeddings = embeddings
        self.vocabulary = vocabulary
        self.reversedVocabulary = dict(zip(vocabulary.values(), vocabulary.keys())) if type(vocabulary) is None else None

    # Method to calculate the embedding matrix
    def calculate(self, dataset=None, *, window_size=1,
                  pmi=True, smoothing: float = 0.0, from_zero=False,
                  d_model=128):

        # Step 1: Calculate the co-occurrence matrix from the dataset
        com_matrix = self.co_occurrence_matrix(dataset, window_size=window_size)
        pmi_matrix = com_matrix

        # Step 2: Calculate PMI (Point wise Mutual Information) if enabled
        if pmi:
            pmi_progressbar = tqdm(pmi_matrix, total=1, desc="Calculating Point wise Mutual Information")
            pmi_matrix = self.pointwise_mutual_information(com_matrix, smoothing=smoothing, from_zero=from_zero)

            pmi_progressbar.update(1)
            pmi_progressbar.close()

        # Step 3: Apply Singular Value Decomposition (SVD) to the PMI matrix
        svd_progressbar = tqdm(pmi_matrix, total=1, desc="Calculating Singular Value Decomposition")
        svd_matrix = self.singular_value_decomposition(pmi_matrix, d_model=d_model)

        svd_progressbar.update(1)
        svd_progressbar.close()

        # Step 4: Create embeddings from the SVD results and map tokens to vectors
        self.embeddings = {token: weights.tolist() for token, weights in
                           tqdm(zip(self.vocabulary, svd_matrix), total=len(self.vocabulary), desc="Processing Tokens")}

        # Adding the End of Sentence token to vocabulary
        self.vocabulary['<EOS>'] = len(self.vocabulary)

        # Create reversed vocabulary (from index to token)
        self.reversedVocabulary = dict(zip(self.vocabulary.values(), self.vocabulary.keys()))

    # Method to calculate co-occurrence matrix from dataset
    def co_occurrence_matrix(self, dataset=None, *, window_size=1):
        # Creating vocabulary from dataset by assigning unique indexes to tokens and initialize co-occurrence matrix
        self.vocabulary = {word: i for i, word in
                           enumerate(sorted(set(token for tokens in dataset for token in tokens)))}
        com_matrix = np.zeros((len(self.vocabulary), len(self.vocabulary)), dtype=int)

        # Iterate over the dataset to fill the co-occurrence matrix
        for tokens in tqdm(dataset, desc="Calculating Co-Occurrence Matrix"):
            for index, token in enumerate(tokens):

                # Get tokens within the specified window size
                tokens_in_window = tokens[max(0, index - window_size): index + window_size + 1]
                tokens_in_window.remove(token)

                # Update co-occurrence matrix for each pair of token and context token
                for context_token in tokens_in_window:
                    com_matrix[self.vocabulary[token], self.vocabulary[context_token]] += 1

        return com_matrix

    # Static method to calculate Point wise Mutual Information (PMI)
    @staticmethod
    def pointwise_mutual_information(com, *, smoothing: float = 0.0, from_zero=False):
        # Total number of co-occurrences across all tokens
        total_co_occurrences = np.sum(com)

        # Prevent division by zero in case of empty co-occurrences
        if total_co_occurrences == 0:
            raise Exception("Number 0 cannot be the value of total_co_occurrences")

        # Calculate token and context probabilities
        token_probability = np.sum(com, axis=1, keepdims=True) / total_co_occurrences
        context_probability = np.sum(com, axis=0, keepdims=True) / total_co_occurrences

        # Calculate the pairwise co-occurrence probability (with smoothing)
        pwc = (com + smoothing) / total_co_occurrences

        # Compute the PMI values with error handling for division by zero or invalid values
        with np.errstate(divide='ignore', invalid='ignore'):
            pmi = np.log(pwc / (token_probability * context_probability))
            pmi[np.isinf(pmi)] = 0

            # Optionally set negative PMI values to zero
            if from_zero:
                pmi[pmi < 0] = 0

        return pmi

    # Static method to apply Singular Value Decomposition (SVD) to PMI matrix
    @staticmethod
    def singular_value_decomposition(pmi, *, d_model=128):
        # Perform SVD decomposition on the PMI matrix
        u, sigma, vt = np.linalg.svd(pmi, full_matrices=False)

        # Reduce dimensionality by keeping only the top 'd_model' components
        u_reduced = u[:, :d_model]
        sigma_reduced = np.diag(sigma[:d_model])

        # Return the reduced representation as the final embedding
        return np.dot(u_reduced, sigma_reduced)

def embed_sequence(sentence, *, embed_dataset=None, forbidden_chars=string.punctuation, token_size=255, max_sequence=12,
                   d_model=128):
    """
    Tokenizes input sentence into smaller chunks (tokens), retrieves embeddings for each token from a provided dataset,
    and returns a padded list of embeddings. If any token is not found in the dataset, an error is raised.

    Args:
        sentence (str): The input sentence to be tokenized and embedded.
        embed_dataset (dict, optional): A dictionary mapping tokens to their corresponding embeddings.
        forbidden_chars (str, optional): A string of characters that should be excluded during tokenization (default is punctuation).
        token_size (int, optional): The maximum size for each token (default is 255).
        max_sequence (int, optional): The maximum length of the sequence to return (default is 12).
        d_model (int, optional): The dimensionality of the embedding vector (default is 128).

    Returns:
        list or None: A list of embeddings corresponding to the tokenized sentence, padded to the specified maximum sequence length,
                        or None if an unknown token is encountered.

    1. The function uses a simple tokenizer to break the input sentence into tokens.
    2. It then attempts to retrieve embeddings for each token from the provided `embed_dataset`.
    3. If the number of embeddings is less than `max_sequence`, zero vectors are used to pad the result.
    4. If any token is not found in the `embed_dataset`, the function prints an error message and returns `None`.
    5. The final embedding list has a length equal to `max_sequence`, with each element being a vector of length `d_model`.
    """

    # Tokenize the input sentence
    tokens = simple_tokenizer(sentence, token_size=token_size, forbidden_chars=forbidden_chars)

    try:

        # Attempt to retrieve embeddings for each token in the tokenized sentence
        # If there are fewer tokens than the maximum allowed sequence length, pad the embeddings with zeros
        embed = [embed_dataset[token] for token in tokens[0]]
        embed += [[0] * d_model] * (max_sequence - len(embed))

        return embed

    except KeyError:

        # In case of an unknown token (not in embed_dataset), print an error message
        print(f"Unknown word has occurred: {sentence}")

    return None

def token_analysis(tokens):
    """
    Analyzes a list of text tokens and calculates various statistics about their lengths and complexity.

    Args:
        tokens (list): List of string tokens to analyze
    """

    # Basic token count metrics
    num_tokens = len(tokens)                                    # Total number of tokens
    total_token_length = sum(len(token) for token in tokens)    # Combined length of all tokens
    avg_token_length = 0                                        # Initialize average length

    # Calculate average length safely (avoid division by zero)
    if num_tokens > 0:
        avg_token_length = total_token_length / num_tokens  # Mean token length

    # Rounding metrics for practical applications
    avg_token_length_ceil = math.ceil(avg_token_length)     # Rounded-up average length

    # Print basic statistics
    output_string = (f'\n\033[37mNumber of tokens:\033[0m {num_tokens}\n'
                     f'\033[37mTotal token length:\033[0m {total_token_length}\n'
                     f'\033[37mAverage token length:\033[0m {avg_token_length}\n'
                     f'\033[37mCeiling of average token length:\033[0m {avg_token_length_ceil}\n')

    # Complexity analysis (requires at least 2 tokens)
    if num_tokens > 1:
        # Calculate variance of token lengths
        variance = sum((len(token) - avg_token_length) ** 2 for token in tokens) / num_tokens

        # Use standard deviation as complexity score (sqrt of variance)
        output_string += f'\033[37mComplexity Score (Standard Deviation of Token Lengths):\033[0m {math.sqrt(variance):.2f}'
    else:
        # Handle case with insufficient data points for variance calculation
        output_string += '\033[37mComplexity Score: Not enough tokens to calculate variance\033[0m'

    return output_string

# Embed Processing

def encode_sentence(sentence, *, embed_dataset=None, forbidden_chars=string.punctuation, token_size=255,
                    max_sequence=12, d_model=2):
    """
    Tokenizes input sentence into smaller chunks, embeds them, applies positional encoding, and adds self-awareness.

    Args:
        sentence (str): The input sentence to encode.
        embed_dataset (optional): The dataset to use for embedding the sentence. Defaults to None.
        forbidden_chars (str): Characters to remove from the sentence before tokenizing. Defaults to string.punctuation.
        token_size (int): The maximum number of tokens per chunk. Defaults to 255.
        max_sequence (int): The maximum sequence length of the encoded sentence. Defaults to 12.
        d_model (int): The dimensionality of the embedding space. Defaults to 2.

    Returns:
        numpy.ndarray: A self-aware sentence representation after embedding, positional encoding, and self-attention.

    1. Embeds the input sentence into a numerical format using the provided embedding dataset.
    2. Applies positional encoding to maintain word order in the sentence.
    3. Adds self-awareness to the sentence using self-attention mechanisms.
    4. Returns the final sentence representation that incorporates all of these features.
    """

    # Embedding the sentence
    embed_ = np.array(
        embed_sequence(sentence, embed_dataset=embed_dataset, forbidden_chars=forbidden_chars, token_size=token_size,
                       max_sequence=max_sequence, d_model=d_model))

    # Applying simple positional encoding to the embedded sentence for order awareness
    # Adding simple "self-awareness" to the sentence using the embedded sentence and positional encoding
    spe = simple_positional_encoder(embed_)
    ssa = simple_self_awareness(embed_ + spe)

    return ssa

def simple_self_awareness(embedded):
    """
    Applies a self-awareness mechanism to the embedded input data by computing self-attention-like operations.

    Args:
        embedded (numpy.ndarray): The input embedded sentence or data. Each row represents an embedded token or element.

    Returns:
        numpy.ndarray: The transformed self-aware output after applying the self-attention mechanism.

    1. Computes a similarity matrix using the dot product of the embedded input with its transpose.
    2. Applies a normalization step using the softmax function along the rows.
    3. Recomputes the output by applying the normalized similarity matrix to the original embeddings.
    4. Returns the self-aware representation of the embedded input.
    """

    # Compute the similarity matrix by taking the dot product of the embedded input with its transpose
    ssa_out = np.dot(embedded, embedded.T)

    # Normalization by applying softmax function
    ssa_out = np.exp(ssa_out - np.max(ssa_out, axis=1, keepdims=True))
    ssa_out = ssa_out / np.sum(ssa_out, axis=1, keepdims=True)

    # Apply the normalized similarity matrix to the embedded input to get the self-aware output
    ssa_out = np.dot(ssa_out, embedded)

    # Return the self-aware output
    return ssa_out

def simple_positional_encoder(tensor):
    """
    Adds positional encoding to the input tensor to provide information about token positions.

    Args:
        tensor (numpy.ndarray): The input tensor representing the embedded tokens. Its shape should be (sequence_length, embedding_dim).

    Returns:
        numpy.ndarray: A tensor with added positional encoding. The same shape as the input tensor.

    1. Computes the positional encodings for each position in the input sequence.
    2. For even dimensions, applies sine function to encode positions.
    3. For odd dimensions, applies cosine function to encode positions.
    4. Combines both encodings to produce a final positional encoding tensor.
    5. Returns the tensor with the added positional encodings.
    """

    # Get the number of positions (sequence length) and the embedding dimension
    pos = tensor.shape[0]
    d_model = tensor.shape[1]

    # Calculate the denominators for sine and cosine positional encodings
    even_i = np.arange(0, d_model, 2)
    e_denominator = np.pow(10000, even_i / d_model)

    odd_i = np.arange(1, d_model, 2)
    o_denominator = np.pow(10000, (odd_i - 1) / d_model)

    # Create an array representing positions in the sequence
    position = np.arange(pos).reshape(pos, 1)

    # Apply sine for even dimensions and cosine for odd dimensions
    even_pe = np.sin(position / e_denominator)
    odd_pe = np.cos(position / o_denominator)

    # Stack and reshape the sine and cosine encodings
    stacked = np.stack([even_pe, odd_pe], axis=1)
    stacked = stacked.reshape(stacked.shape[0], -1, order='F')

    # Return the tensor with positional encoding
    return stacked

# AI

def he_initialization(n_inputs, n_neurons):
    return np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)

class LayerDense:
    def __init__(self, n_neurons, a_function, w_init=he_initialization,
                 dropout=False, d_rate=0.1,
                 regularizers=(0, 0, 0, 0),
                 **a_func_kwargs):

        """
            Dense (fully-connected) neural network layer with optional features

            Parameters:
            n_neurons (int): Number of neurons in the next layer
            a_function (function): Activation function to use
            w_init (function): Weight initialization function (default: He initialization)
            dropout (bool): Whether to use dropout (default: False)
            d_rate (float): Dropout rate (probability of dropping, converted to keep probability)
            regularizers (tuple): Regularization coefficients (w_l1, w_l2, b_l1, b_l2)
            **a_func_kwargs: Additional arguments for activation function
        """

        # Initialization parameters
        self.w_init = w_init            # Weight initialization method
        self.kwargs = a_func_kwargs     # Extra arguments for activation function
        self.a_function = a_function    # Activation function

        # Layer parameters
        self.weights = None                         # Weight matrix (to be initialized)
        self.biases = np.zeros((1, n_neurons))      # Bias vector

        # Dropout configuration
        self.dropout = dropout      # Whether to apply dropout
        self.d_rate = 1 - d_rate    # Convert dropout rate to keep probability

        # Regularization coefficients (L1, L2) for weights and biases
        self.w_regularizer = (regularizers[0], regularizers[1])     # (L1, L2) for weights
        self.b_regularizer = (regularizers[2], regularizers[3])     # (L1, L2) for biases

        # Optimization parameters (for momentum-based optimizers)
        self.w_b_momentums = [None, np.zeros_like(self.biases)]     # Weight and bias momentums
        self.w_b_cache = [None, np.zeros_like(self.biases)]         # Cache for adaptive optimizers (e.g., Adam)

        # Layer state storage
        self.layer_output = None        # Pre-activation output (Wx + b)
        self.activated_output = None    # Post-activation output
        self.dropout_output = None      # Dropout mask
        self.inputs = None              # Input values from previous layer
        self.output = None              # Final output after activation and dropout

    def forward(self, inputs):
        # Ensure inputs are at least 2D array
        self.inputs = np.atleast_2d(np.array(inputs))

        # Initialize weights if first forward pass
        if self.weights is None:
            self.weights = self.w_init(self.inputs.shape[-1], self.biases.shape[-1])
            self.w_b_momentums[0] = np.zeros_like(self.weights)
            self.w_b_cache[0] = np.zeros_like(self.weights)

        # Calculate linear transformation: Wx + b
        self.layer_output = np.dot(self.inputs, self.weights) + self.biases

        # Apply activation function
        self.activated_output = self.a_function(
            **{**self.kwargs, 'derivative': False, 'layer_output': self.layer_output.copy()}
        )

        # Apply dropout if enabled
        if self.dropout:
            # Create binary mask and scale by keep probability
            self.dropout_output = np.random.binomial(1, self.d_rate, size=self.activated_output.shape) / self.d_rate
            self.output = self.activated_output * self.dropout_output
        else:
            self.output = self.activated_output

    def backward(self, dvalues):
        # Apply dropout mask to gradients if dropout was used
        derrived_d = dvalues * self.dropout_output if self.dropout else dvalues.copy()

        # Calculate derivative of activation function
        derrived_a_f = self.a_function(
            **{**self.kwargs, 'derivative': True, 'layer_output': self.layer_output.copy(),
               'activated_output': self.activated_output.copy(), 'derrived_values': derrived_d}
        )

        # Calculate gradients for weights, biases, and inputs
        self.dweights = np.dot(self.inputs.T, derrived_a_f)             # Gradient for weights
        self.dbiases = np.sum(derrived_a_f, axis=0, keepdims=True)      # Gradient for biases
        self.dinputs = np.dot(derrived_a_f, self.weights.T)             # Gradient for inputs (to previous

        # Weight regularization
        if self.w_regularizer[0] > 0:  # L1 regularization
            self.dweights += self.w_regularizer[0] * np.sign(self.weights)
        if self.w_regularizer[1] > 0:  # L2 regularization
            self.dweights += self.w_regularizer[1] * 2 * self.weights

        # Bias regularization
        if self.b_regularizer[0] > 0:  # L1 regularization
            self.dbiases += self.b_regularizer[0] * np.sign(self.biases)
        if self.b_regularizer[1] > 0:  # L2 regularization
            self.dbiases += self.b_regularizer[1] * 2 * self.biases

def activation_leaky_relu(*, alpha=0.1, derivative=False, layer_output=None, activated_output=None, derrived_values=None):
    """
        Leaky ReLU activation function implementation with forward and backward passes

        Parameters:
        alpha (float): Slope for negative values (default: 0.1)
        derivative (bool): Flag to calculate derivative instead of activation
        layer_output: Pre-activation values from layer (Z = Wx + b)
        activated_output: Not used in Leaky ReLU, maintained for API consistency
        derrived_values: Upstream gradients from next layer (for backward pass)
    """

    if not derivative:
        # Forward pass - Leaky ReLU activation
        # Element-wise operation: f(x) = x if x > 0 else alpha * x
        return np.maximum(alpha * layer_output, layer_output)

    else:
        # Backward pass - Calculate derivative
        # Create gradient matrix initialized to 1s (positive slope)
        drelu = np.ones_like(layer_output)

        # Where input values were <= 0, set gradient to alpha (leaky slope)
        drelu[layer_output <= 0] = alpha

        # Multiply upstream gradients by local derivative (chain rule)
        derrived_values *= drelu

        return derrived_values

def activation_softmax(*, derivative=False, layer_output=None, activated_output=None, derrived_values=None):
    """
    Softmax activation function with stable numerical implementation and gradient calculation

    Parameters:
    derivative (bool): True for backward pass, False for forward pass
    layer_output: Pre-activation values from dense layer (Z = Wx + b)
    activated_output: Output from forward pass (required for gradient calculation)
    derrived_values: Upstream gradients from next layer (for backward pass)
    """

    if not derivative:
        # Forward pass implementation with numerical stability
        # Subtract max value to prevent exponential overflow
        exp_values = np.exp(layer_output - np.max(layer_output, axis=1, keepdims=True))

        # Normalize to create probability distribution (sums to 1 per row)
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    else:
        # Backward pass calculation for softmax gradient
        # Initialize container for final gradients
        derrived_inputs = np.empty_like(derrived_values)

        # Calculate Jacobian matrix for each sample in batch
        for idx, (single_output, single_dvalues) in enumerate(zip(activated_output, derrived_values)):
            # Reshape to column vector for matrix operations
            single_output = single_output.reshape(-1, 1)

            # Create Jacobian matrix using identity: J = diag(S) - S.S^T
            jacobian = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            # Multiply Jacobian with upstream gradients (chain rule)
            derrived_inputs[idx] = np.dot(jacobian, single_dvalues)

        return derrived_inputs

class OptimizerAdam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        """
        Adam (Adaptive Moment Estimation) optimizer implementation

        Parameters:
        learning_rate (float): Initial step size (default: 0.001)
        decay (float): Learning rate decay rate (default: 0 - no decay)
        epsilon (float): Small value to prevent division by zero (default: 1e-7)
        beta_1 (float): Exponential decay rate for first moment estimates (default: 0.9)
        beta_2 (float): Exponential decay rate for second moment estimates (default: 0.999)
        """

        # Initialize optimizer parameters
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate      # With decay applied
        self.decay = decay
        self.iterations = 0                             # Track update counts for decay and bias correction
        self.epsilon = epsilon
        self.beta_1 = beta_1                            # For momentum (first moment)
        self.beta_2 = beta_2                            # For cache (second moment)

    def pre_update_params(self):
        # Update learning rate with decay before parameter updates
        if self.decay:
            # Time-based learning rate decay: lr = initial_lr / (1 + decay * iteration)
            self.current_learning_rate = self.learning_rate / (1. + self.decay * self.iterations)

    def update_params(self, layer):
        # Update layer parameters using Adam optimization algorithm
        # Calculate momentum for weights and biases (moving average of gradients)
        layer.w_b_momentums[0] = self.beta_1 * layer.w_b_momentums[0] + (1 - self.beta_1) * layer.dweights
        layer.w_b_momentums[1] = self.beta_1 * layer.w_b_momentums[1] + (1 - self.beta_1) * layer.dbiases

        # Compute bias-corrected momentums (counteract initial zero-bias)
        weight_momentums_corrected = layer.w_b_momentums[0] / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.w_b_momentums[1] / (1 - self.beta_1 ** (self.iterations + 1))

        # Update cache with squared gradients (moving average of squared gradients)
        layer.w_b_cache[0] = self.beta_2 * layer.w_b_cache[0] + (1 - self.beta_2) * layer.dweights ** 2
        layer.w_b_cache[1] = self.beta_2 * layer.w_b_cache[1] + (1 - self.beta_2) * layer.dbiases ** 2

        # Compute bias-corrected cache estimates
        weight_cache_corrected = layer.w_b_cache[0] / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.w_b_cache[1] / (1 - self.beta_2 ** (self.iterations + 1))

        # Update parameters with adaptive learning rate
        layer.weights -= self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases -= self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        # Increment iteration counter after parameter updates
        self.iterations += 1

class Loss:
    def regularization_loss(self, layer):
        """
        Calculate L1 and L2 regularization loss for a layer's weights and biases

        Parameters:
        layer: Layer object containing weights/biases and their regularization terms
        """

        regularization_loss = 0

        # Process both weights and biases with their respective regularizers
        for regularizer, param in [(layer.w_regularizer, layer.weights), (layer.b_regularizer, layer.biases)]:

            # L1 regularization (Lasso)
            if regularizer[0] > 0:      # Check if L1 regularization is enabled
                regularization_loss += regularizer[0] * np.sum(np.abs(param))

            # L2 regularization (Ridge)
            if regularizer[1] > 0:      # Check if L2 regularization is enabled
                regularization_loss += regularizer[1] * np.sum(param ** 2)

        return regularization_loss

    def calculate(self, output, y):
        """
        Calculate mean loss across all samples in batch

        Parameters:
        output: Model predictions
        y: Ground truth/target values
        """

        # Get individual losses for each sample in the batch
        sample_losses = self.forward(output, y)

        # Return mean loss for the entire batch
        return np.mean(sample_losses)

class LossCategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        """
        Compute categorical cross-entropy loss

        Parameters:
        y_pred: Model predictions (probabilities from softmax)
        y_true: Ground truth labels (integer indices or one-hot encoded)
        """

        # Number of samples in batch
        samples = len(y_pred)

        # Clip predictions to prevent log(0) errors
        # Maintains numerical stability with 1e-7 margin
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate losses only for true class indices
        # For one-hot labels: y_true would need to be argmax first
        return -np.log(y_pred_clipped[range(samples), y_true])

    def backward(self, dvalues, y_true):
        """
        Compute gradient of loss with respect to inputs

        Parameters:
        dvalues: Values from subsequent layer (gradients)
        y_true: Ground truth labels (integer indices or one-hot encoded)
        """

        samples = len(dvalues)              # Number of samples
        num_classes = dvalues.shape[1]      # Number of classes

        # Convert sparse labels to one-hot vectors if needed
        if len(y_true.shape) == 1:                      # Sparse labels (class indices)
            y_true = np.eye(num_classes)[y_true]        # Create one-hot encoding

        # Compute gradient of loss with respect to inputs
        # Normalize gradient by number of samples to maintain scale
        self.dinputs = -y_true / dvalues / samples

        # For softmax activation combined with cross-entropy loss,
        # this simplifies to (y_pred - y_true) when chained together
        # The current implementation handles general case

class ActivationSoftmaxLossCategoricalCrossentropy:
    def __init__(self):
        # Combined Softmax activation and Categorical Cross-Entropy loss for efficiency
        # Initialize separate loss calculator
        self.loss = LossCategoricalCrossentropy()
        self.output = None          # Will store softmax probabilities
        self.dinputs = None         # Will store gradients

    def forward(self, inputs, y_true):
        """
        Forward pass combining softmax activation and cross-entropy loss

        Parameters:
        inputs: Pre-activation values from final dense layer
        y_true: Ground truth labels (integer indices)
        """

        # Apply softmax activation to create probability distribution
        self.output = activation_softmax(layer_output=inputs)

        # Calculate and return mean cross-entropy loss
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        """
        Backward pass with optimized gradient calculation

        Parameters:
        dvalues: Unused parameter (kept for interface consistency)
        y_true: Ground truth labels (integer indices)
        """

        # Number of samples in batch
        samples = len(dvalues)

        # Create copy of gradients to modify (start with softmax derivative)
        self.dinputs = dvalues.copy()

        # Apply simplified gradient formula for combined softmax + cross-entropy:
        # gradient = y_pred - y_true
        # For true class indices, subtract 1 (equivalent to y_pred - 1)
        self.dinputs[range(samples), y_true] -= 1

        # Normalize gradients by number of samples (average across batch)
        self.dinputs /= samples

        return self.dinputs

# Model hyperparameters
d_mod = 256     # Dimension of embeddings
max_seq = 32    # Maximum sequence length for inputs

if __name__ == "__main__":
    word_dataset = None                 # Initialize word dataset
    embed_dataset = EmbeddingData()     # Create embedding database object

    # Define neural network architecture with dropout regularization
    dense_layers = [LayerDense(512, activation_leaky_relu, dropout=True, d_rate=0.05),
                    LayerDense(256, activation_leaky_relu, dropout=True, d_rate=0.05),
                    LayerDense(512, activation_leaky_relu, dropout=True, d_rate=0.05),
                    LayerDense(10, activation_leaky_relu, dropout=True, d_rate=0.05)]

    # Load or create embeddings
    inputted = input('Do you want load pre-existing EmbeddingData and word dataset ? (Y/N):\n')

    if inputted.upper() == 'Y':
        # Load saved embeddings from file
        with open('embeddings_database.json', 'r') as file:
            embeddings_database = json.load(file)

            embed_dataset.embeddings = embeddings_database['embeddings']
            embed_dataset.vocabulary = embeddings_database['vocabulary']

    else:
        # Create new embeddings from word dataset
        with open('word_dataset.json', 'r') as file:
            word_dataset = json.load(file)['word_dataset']

        # Calculate embeddings using tokenized data
        embed_dataset.calculate(simple_tokenizer(word_dataset), smoothing=3e-4, d_model=d_mod, window_size=3)

    # Model weights loading or training
    inputted = input('Do you want load pre-existing weights and biases thus skip training ? (Y/N):\n')

    if inputted.upper() == 'Y':
        # Load saved model parameters
        with open('model_data.json', 'r') as file:
            model_data = json.load(file)

            for idx in range(len(dense_layers)):
                dense_layers[idx].weights = model_data[f'w_{idx}']
                dense_layers[idx].biases = model_data[f'b_{idx}']

    else:
        # Training configuration
        with open('training_data.json', 'r') as file:
            training_data = json.load(file)

            training_x = training_data[f'training_data']
            training_y = training_data[f'truth']

        # Prepare training data (convert sentences to embeddings)
        X = np.array([encode_sentence(x, embed_dataset=embed_dataset.embeddings, max_sequence=max_seq, d_model=d_mod).flatten() for x in training_x])
        y = np.array(training_y)

        # Initialize loss and optimizer
        loss_activation = ActivationSoftmaxLossCategoricalCrossentropy()
        optimizer = OptimizerAdam(learning_rate=0.005, decay=3e-4)

        # Training loop with progress bar
        training_loop = tqdm(range(1001))
        for epoch in training_loop:

            # Forward pass through all layers
            dense_layers[0].forward(X)
            for idx in range(1, len(dense_layers)):
                dense_layers[idx].forward(dense_layers[idx-1].output)

            # Calculate loss and accuracy
            loss = loss_activation.forward(dense_layers[len(dense_layers) - 1].output, y)
            predictions = np.argmax(loss_activation.output, axis=1)
            accuracy = np.mean(predictions == y)

            # Update progress bar every 100 epochs
            if not epoch % 100:
                training_loop.desc = f'Training Progress: | Acc: {accuracy} Loss: {loss} Lr: {optimizer.current_learning_rate} |'

            # Backpropagation through network layers
            dense_layers[-1].backward(loss_activation.backward(loss_activation.output, y))
            for idx in range(0, len(dense_layers)-1)[::-1]:
                dense_layers[idx].backward(dense_layers[idx+1].dinputs)

            # Update parameters with optimizer
            optimizer.pre_update_params()
            for layer in dense_layers:
                optimizer.update_params(layer)
            optimizer.post_update_params()

        # Save trained weights option
        inputted = input(f'Do you want to save new weights and biases ? (Y/N)')

        if inputted.upper() == 'Y':
            json_m_object = json.dumps({"w_0": dense_layers[0].weights.tolist(),
                                            "w_1": dense_layers[1].weights.tolist(),
                                            "w_2": dense_layers[2].weights.tolist(),
                                            "w_3": dense_layers[3].weights.tolist(),
                                            "b_0": dense_layers[0].biases.tolist(),
                                            "b_1": dense_layers[1].biases.tolist(),
                                            "b_2": dense_layers[2].biases.tolist(),
                                            "b_3": dense_layers[3].biases.tolist()}, indent=4)

            with open('model_data.json', 'w') as file:
                file.write(json_m_object)

    # Disable dropout for inference
    for obj in dense_layers:
        obj.dropout = False

    # Question type classification labels
    question_types = [
        'temperature', 'time of the day', 'date', 'name', 'surname',
        'hobby', 'eating', 'movie', 'book', 'animal'
    ]

    # Interactive prediction loop
    while True:
        inputted = input('\nInput sentence in polish:\n')
        proceed = True

        # Tokenize user input using simple_tokenizer and extract the token list
        # Perform detailed analysis on token statistics including count, lengths, and complexity metrics
        tokens = simple_tokenizer(inputted)[0]

        # Clipping token size to max_seq
        if len(tokens) > max_seq:
            tokens = tokens[0:max_seq]

        # Analyze tokens
        print(token_analysis(tokens))

        # Check for unknown words in input
        for word in tokens:
            if word not in embed_dataset.vocabulary:
                print(f'Word: {word} not in the existing vocabulary. Please add it to the database')
                proceed = False

        # Stops the loop from proceeding due to the unknown word in the input message
        if not proceed:
            continue

        # Process input through the network
        dense_layers[0].forward(encode_sentence(inputted, embed_dataset=embed_dataset.embeddings, max_sequence=max_seq, d_model=d_mod).flatten())
        for idx in range(1, len(dense_layers)):
            dense_layers[idx].forward(dense_layers[idx - 1].output)

        # Get final prediction probabilities
        probability_distribution = activation_softmax(layer_output=dense_layers[-1].output)
        print(f'\n\033[93m{inputted}:\033[0m You asked about the {question_types[np.argmax(probability_distribution)]}')