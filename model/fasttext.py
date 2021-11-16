import torch.nn as nn
import torch.nn.functional as F

class FastText(nn.Module):
    def __init__(self, input_dim, args):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = args.HIDDEN_DIM
        self.output_dim = args.OUTPUT_DIM

        #maps sparse vector inputs to dense vector word embeddings
        self.embedding = nn.Embedding(input_dim, self.hidden_dim)

        #goes from the average of the word embeddings to the output prediction
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        #x = [batch size, seq. length]


        x = self.embedding(x)
        #x = [batch size, seq. length, hidden dim]

        x = F.avg_pool2d(x, (x.shape[1], 1)).squeeze(1) #averages word vectors across whole sequence length
        #x = [batch size, hidden dim]

        x = self.fc(x)
        #x = [batch size, output dim]

        return x