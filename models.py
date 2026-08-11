import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.w)
    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        score = self.run(x)
        scoreScalar = nn.as_scalar(score)
        if scoreScalar >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        while True:
            mistake = False
            for x, y in dataset.iterate_once(1):
                label = nn.as_scalar(y)
                prediction = self.get_prediction(x)
                if prediction != label:
                    self.w.update(x, label)
                    mistake = True
            if not mistake:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # One-hidden-layer MLP to approximate sin(x): 1 -> 512 -> 1
        hiddenSize = 512
        # Hidden layer parameters (input -> hidden)
        self.w1 = nn.Parameter(1, hiddenSize)
        self.b1 = nn.Parameter(1, hiddenSize)
        # Output layer parameters (hidden -> output)
        self.w2 = nn.Parameter(hiddenSize, 1)
        self.b2 = nn.Parameter(1, 1)
        self.lr = 0.05

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        # Hidden layer: linear + bias + ReLU
        l1Linear = nn.Linear(x, self.w1)
        l1Biased = nn.AddBias(l1Linear, self.b1)
        l1Activated = nn.ReLU(l1Biased)
        # Output layer: linear + bias (no ReLU so outputs can be negative)
        l2Linear = nn.Linear(l1Activated, self.w2)
        return nn.AddBias(l2Linear, self.b2)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        # Mean squared error between predictions and targets
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        # Gradient descent until full-dataset loss is at most 0.02
        while True:
            for x, y in dataset.iterate_once(200):
                loss = self.get_loss(x, y)
                parameters = [self.w1, self.b1, self.w2, self.b2]
                gradient = nn.gradients(loss, parameters)
                # Negative learning rate => gradient descent (not ascent)
                self.w1.update(gradient[0], -self.lr)
                self.b1.update(gradient[1], -self.lr)
                self.w2.update(gradient[2], -self.lr)
                self.b2.update(gradient[3], -self.lr)
            # Check once per epoch on all data (matches autograder metric)
            if nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))) <= 0.02:
                return

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # One-hidden-layer MLP for MNIST: 784 -> 200 -> 10
        hiddenSize = 200
        # Hidden layer parameters (flattened image -> hidden)
        self.w1 = nn.Parameter(784, hiddenSize)
        self.b1 = nn.Parameter(1, hiddenSize)
        # Output layer parameters (hidden -> 10 digit class logits)
        self.w2 = nn.Parameter(hiddenSize, 10)
        self.b2 = nn.Parameter(1, 10)
        self.lr = 0.5

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        # Hidden layer: linear + bias + ReLU
        l1Linear = nn.Linear(x, self.w1)
        l1Biased = nn.AddBias(l1Linear, self.b1)
        l1Activated = nn.ReLU(l1Biased)
        # Output layer: linear + bias (no ReLU on logits)
        l2Linear = nn.Linear(l1Activated, self.w2)
        return nn.AddBias(l2Linear, self.b2)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        # Softmax cross-entropy over the 10 digit classes
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        # Gradient descent until validation accuracy is safely above 97%
        while True:
            for x, y in dataset.iterate_once(200):
                loss = self.get_loss(x, y)
                parameters = [self.w1, self.b1, self.w2, self.b2]
                gradient = nn.gradients(loss, parameters)
                # Negative learning rate => gradient descent
                self.w1.update(gradient[0], -self.lr)
                self.b1.update(gradient[1], -self.lr)
                self.w2.update(gradient[2], -self.lr)
                self.b2.update(gradient[3], -self.lr)
            # Stop a bit above 97% so test accuracy is more likely to pass
            if dataset.get_validation_accuracy() >= 0.975:
                return

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # RNN + classifier for language ID over character sequences
        outputSize = len(self.languages)
        hiddenSize = 200
        # Shared recurrent weights: char embedding and previous-hidden transform
        self.wx = nn.Parameter(self.num_chars, hiddenSize)
        self.whidden = nn.Parameter(hiddenSize, hiddenSize)
        self.bhidden = nn.Parameter(1, hiddenSize)
        # Output layer: final hidden state -> language logits
        self.woutput = nn.Parameter(hiddenSize, outputSize)
        self.boutput = nn.Parameter(1, outputSize)
        self.lr = 0.5

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        # First character: no previous hidden state
        z = nn.Linear(xs[0], self.wx)
        z = nn.AddBias(z, self.bhidden)
        h = nn.ReLU(z)
        # Remaining characters: combine current char with previous h (shared weights)
        for i in range(1, len(xs)):
            z = nn.Add(nn.Linear(xs[i], self.wx), nn.Linear(h, self.whidden))
            z = nn.AddBias(z, self.bhidden)
            h = nn.ReLU(z)
        # Classify from the final hidden summary of the word (no ReLU on logits)
        return nn.AddBias(nn.Linear(h, self.woutput), self.boutput)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        # Softmax cross-entropy over the 5 languages
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        # Gradient descent until validation accuracy is safely above 81%
        while True:
            for xs, y in dataset.iterate_once(100):
                loss = self.get_loss(xs, y)
                parameters = [self.wx, self.whidden, self.bhidden, self.woutput, self.boutput]
                gradient = nn.gradients(loss, parameters)
                # Negative learning rate => gradient descent
                self.wx.update(gradient[0], -self.lr)
                self.whidden.update(gradient[1], -self.lr)
                self.bhidden.update(gradient[2], -self.lr)
                self.woutput.update(gradient[3], -self.lr)
                self.boutput.update(gradient[4], -self.lr)
            # Stop above the 81% test threshold to leave some margin
            if dataset.get_validation_accuracy() >= 0.85:
                return