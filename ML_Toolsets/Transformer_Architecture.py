import torch
import torch.nn as nn
from einops import rearrange
import torch.optim as optim
import torch.utils.data as data
from sklearn.model_selection import train_test_split
import math
import time
import numpy as np
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
global architecture_analysis
architecture_analysis = False
global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # if multiple GPUs, use cuda:0, cuda:1, etc.

class MultiHeadAttention(nn.Module):
    ''' Docsrt'''
    def __init__(self, emb_sz, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert emb_sz % num_heads == 0, "emb_sz must be divisible by num_heads"
        self.emb_sz = emb_sz
        self.num_heads = num_heads
        self.d_k = emb_sz // num_heads

        self.W_q = nn.Linear(emb_sz, emb_sz)
        self.W_k = nn.Linear(emb_sz, emb_sz)
        self.W_v = nn.Linear(emb_sz, emb_sz)
        self.W_o = nn.Linear(emb_sz, emb_sz)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, emb_sz = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.emb_sz)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class FeedForwardNN(nn.Module):
    def __init__(self, emb_sz, n_layers=2, hidden_units=64, dropout=0.2, hyperparameters=None):
        super(FeedForwardNN, self).__init__()
        if hyperparameters is not None: 
            n_layers = hyperparameters['n_layers']
            hidden_units = hyperparameters['hidden_units']
        
        layers = []
        # Input layer
        layers.append(nn.Linear(emb_sz, hidden_units))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Dropout(dropout/2))

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_units, emb_sz))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TimeSeriesEncoding(nn.Module):
    ''' 
    TimeSeries Positional Encoding
    Encodes sequence with time information (daily and weekly patterns)
    Attention head can learn time-based patterns in the data.
    '''
    def __init__(self, vec_dim, seq_len):
        super(TimeSeriesEncoding, self).__init__()
        self.d_model = vec_dim
        self.seq_len = seq_len
        
        # # Sequence positional encoding (absolute)
        # pe = torch.zeros(seq_len, vec_dim)
        # position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, vec_dim, 2).float() * -(math.log(10000.0) / vec_dim))
        # pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        # self.register_buffer('pe', pe.unsqueeze(0))
    
    def plot_columns(self, x, pe_hours, columns):
        fig, axs = plt.subplots(len(columns), 1, figsize=(10, 15))
        fig.suptitle('TimeSeries Encoding', fontsize=16)
        
        for i, col in enumerate(columns):
            axs[i].plot(x[0, :, col].detach().cpu().numpy(), label='Before Encoding')
            axs[i].plot(pe_hours[0, :, col].detach().cpu().numpy(), label='After Encoding')
            axs[i].set_title(f'Column {col+1}')
            axs[i].legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_positional_encoding_dot_product(self, seq_len, K_values, emb_sz, encodings):
        plt.figure(figsize=(10, 6))

        dot_products = []
        for K in K_values:
            if 0 <= (seq_len // 2 + K) < seq_len:
                stationary_encoding = encodings[seq_len // 2].view(emb_sz)
                shifted_encoding = encodings[seq_len // 2 + K].view(emb_sz)
                dot_product = torch.dot(stationary_encoding, shifted_encoding).item()
                dot_products.append(dot_product)
            else:
                dot_products.append(0)

        plt.plot(K_values, dot_products, label=f'd_model = {emb_sz}')
        plt.title('Dot Product of Positional Encodings')
        plt.xlabel('K')
        plt.ylabel('Dot Product (Similarity)')
        plt.legend()
        plt.show()

    def forward(self, x, time_hours):
        # get the size of the input tensor
        batch_size, seq_len, emb_sz = x.size()
        # print(f'Batch size: {batch_size}, Sequence length: {seq_len}, Embedding size: {emb_sz}')
        # Create positional encoding based on hours
        div_term = torch.exp(torch.arange(0, emb_sz, 2, device=device).float() * -(math.log(0.0375) / emb_sz))

        pe_hours = torch.zeros_like(x)
        pe_hours[:, :, 0::2] = torch.sin(time_hours.unsqueeze(-1) * div_term[:emb_sz // 2]) * 0.5
        pe_hours[:, :, 1::2] = torch.cos(time_hours.unsqueeze(-1) * div_term[:emb_sz // 2]) * 0.5
        
        if architecture_analysis:
            # Plotting before and after encoding for columns 1-6 of the first sample. needs debugging for use in training
            self.plot_columns(x[0:1], x[0:1] + pe_hours[0:1], columns=list(range(int(emb_sz-4), emb_sz)))
            self.plot_positional_encoding_dot_product(seq_len, list(range(-seq_len//2, seq_len//2)), emb_sz, (pe_hours[0, :, :]))

        return x + pe_hours 


class Attention_eRPE(nn.Module): 
    '''
    @article{ConvTran2023,
    title={Improving position encoding of transformers for multivariate time series classification},
    author={Foumani, Navid Mohammadi and Tan, Chang Wei and Webb, Geoffrey I and Salehi, Mahsa},
    journal={Data Mining and Knowledge Discovery},
    volume={38},
    number={1},
    pages={22--48},
    year={2024},
    publisher={Springer}
    }
    '''
    def __init__(self, emb_size, num_heads, seq_len, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)
        self.relative_bias_table = nn.Parameter(torch.zeros((2 * self.seq_len - 1), num_heads))
        coords = torch.meshgrid((torch.arange(1), torch.arange(self.seq_len)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[1] += self.seq_len - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(emb_size)
        self.to_out = nn.LayerNorm(emb_size)
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # k,v,q shape = (batch_size, num_heads, seq_len, d_head)
        attn = torch.matmul(q, k) * self.scale
        attn = nn.functional.softmax(attn, dim=-1) # attn shape (seq_len, seq_len)
        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(0, self.relative_index.repeat(1, self.num_heads))
        relative_bias = rearrange(relative_bias, '(h w) c -> 1 c h w', h=1 * self.seq_len, w=1 * self.seq_len)
        attn = attn + relative_bias
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2) # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.reshape(batch_size, seq_len, -1) # out.shape == (batch_size, seq_len, num_heads, d_head)
        # out = self.to_out(out) # out.shape == (batch_size, seq_len, emb_sz)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, emb_sz, num_heads, d_ff, nl_ff, seq_len, dropout):
        super(EncoderLayer, self).__init__()
        # self.self_attn = MultiHeadAttention(emb_sz, num_heads)
        self.self_attn = Attention_eRPE(emb_sz, num_heads, seq_len, dropout)
        self.feed_forward = FeedForwardNN(emb_sz, n_layers=nl_ff, hidden_units=d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(emb_sz)
        self.norm2 = nn.LayerNorm(emb_sz)
        self.dropout = nn.Dropout(dropout/2)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x)
        # attn_output = self.self_attn(x, x, x, mask)
        x = (x + (attn_output)) 
        # x = (x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        # x = (self.dropout(ff_output))
        x = self.norm2(self.dropout(x + ff_output))
        return x


class Attentive_FFNN(nn.Module):
    def __init__(self, config):
        self.input_dim = config["input_dim"]
        self.output_dim = config["output_dim"]
        self.emb_sz = config["emb_sz"]
        self.num_heads = config["num_heads"]
        self.num_layers = config["num_layers"]
        self.d_ff = config["d_ff"]
        self.nl_ff = config["nl_ff"]
        self.seq_len = config["max_seq_length"]
        self.dropout = config["dropout"]
        super(Attentive_FFNN, self).__init__()
        
        self.encoder_input_linear = nn.Linear(self.input_dim, self.emb_sz)
        # self.init_nn = FeedForwardNN(self.emb_sz, n_layers=self.nl_ff, hidden_units=self.d_ff, dropout=self.dropout)
        # self.init_nn = torch.nn.RNN(self.emb_sz, self.emb_sz, num_layers=1, nonlinearity='tanh', dropout=self.dropout)
        # self.embed_layer = nn.Sequential(nn.Conv2d(1, self.emb_sz*2, kernel_size=[1, 3], padding='same'),
        #                  nn.BatchNorm2d(self.emb_sz*2),
        #                  nn.ReLU())
        # self.embed_layer2 = nn.Sequential(nn.Conv2d(self.emb_sz*2, self.emb_sz, kernel_size=[self.input_dim, 1], padding='valid'),
        #                   nn.BatchNorm2d(self.emb_sz),
        #                   nn.ReLU())
        self.time_positional_encoding = TimeSeriesEncoding(self.input_dim, self.seq_len)
        self.encoder_layers = nn.ModuleList([EncoderLayer(self.emb_sz, self.num_heads, self.d_ff, self.nl_ff, self.seq_len, self.dropout) for _ in range(self.num_layers)])
        self.gap = nn.AdaptiveAvgPool1d(1)        
        self.flatten = nn.Flatten()
        self.final_nn = FeedForwardNN(self.emb_sz, n_layers=self.nl_ff, hidden_units=self.d_ff, dropout=self.dropout)
        # self.final_nn = torch.nn.RNN(self.emb_sz, self.emb_sz, num_layers=24, nonlinearity='tanh', dropout=self.dropout)
        self.fc_out = nn.Linear(self.emb_sz, self.output_dim)
        self.dropout = nn.Dropout(self.dropout/2)

    def forward(self, src, tgt, src_mask, tgt_mask):
        time_hours = src[:, :, 0]
        src_embedded = self.encoder_input_linear(src)
        # src_embedded, _ = self.init_nn(src_embedded)
        # src_embedded = self.init_nn(src_embedded)
        # src = src.permute(0, 2, 1)
        # src = src.unsqueeze(1) # Add channel dimension
        # src_embedded = self.embed_layer(src)
        # src_embedded = self.embed_layer2(src_embedded)
        # src_embedded = src_embedded.squeeze(2)
        # src_embedded = src_embedded.permute(0, 2, 1)
        src_encoded = self.time_positional_encoding(src_embedded, time_hours)
        enc_output = src_encoded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        # out = enc_output.permute(0, 2, 1)
        # out = self.gap(out)
        # out = out.permute(0, 2, 1)
        output_nn = self.final_nn(enc_output)
        output = self.fc_out(output_nn)
        return output
    
def count_parameters(model: torch.nn.Module) -> int:
    # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
    """ Returns the number of learnable parameters for a PyTorch model """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Custom Dataset to handle varying-length sequences
class TimeSeriesDataset(data.Dataset):
    def __init__(self, data, target):
        self.data = [torch.tensor(d.values, dtype=torch.float32) for d in data]
        self.target = [torch.tensor(t.values, dtype=torch.float32) for t in target]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

    @staticmethod
    def collate_fn(batch):
        data, target = zip(*batch)
        max_length = max(seq.size(0) for seq in data)
        padded_data = pad_sequences(data, max_length)
        padded_target = pad_sequences(target, max_length)
        return padded_data, padded_target

class EarlyStopping:
    def __init__(self, patience=15, verbose=False, delta=0.0009):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        self.best_loss = val_loss
        torch.save(model.state_dict(), 'Xfrmr_checkpoint.pt')

# Function to generate model predictions
def predict(model, src_batch, max_seq_length, device='cpu', start_token=None):
    model.eval()
    src_batch = src_batch.to(device)
    src_mask, _ = generate_masks(src_batch, src_batch)
    src_mask = src_mask.to(device)
    batch_size = src_batch.size(0)
    tgt_dim = model.fc_out.out_features

    # Initialize decoder input with zeros (or use a start token)
    if start_token == None:
        tgt_input = torch.zeros(batch_size, 1, tgt_dim).to(device)
    else:
        tgt_input = start_token.repeat(batch_size, 1, 1).to(device)
    
    outputs = []

    for _ in range(max_seq_length):
        tgt_mask = (tgt_input.sum(dim=-1) != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = tgt_mask & (1 - torch.triu(torch.ones((1, tgt_input.size(1), tgt_input.size(1)), device=tgt_input.device), diagonal=1)).bool()

        output = model(src_batch, tgt_input, src_mask, tgt_mask)
        outputs.append(output[:, -1:, :].cpu().numpy())

        # Append the latest output to tgt_input for the next step
        tgt_input = torch.cat([tgt_input, output[:, -1:, :]], dim=1)

    outputs = np.concatenate(outputs, axis=1)
    return outputs

def model_params_match(model, Xfrmr_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # if multiple GPUs, use cuda:0, cuda:1, etc.
    checkpoint = torch.load(Xfrmr_path, map_location=device)
    model_state_dict = model.state_dict()
    # Compare the shapes of the parameters
    for name, param in checkpoint.items():
        if name not in model_state_dict:
            return False
        if model_state_dict[name].shape != param.shape:
            return False
    return True

def fitness_func(ga_instance, solution, sol_idx):
    global train_dataloader, torch_ga, model
    model_weights_dict = pygad.torchga.model_weights_as_dict(model=model, weights_vector=solution)
    model.load_state_dict(model_weights_dict)
    model.train()
    start_time = time.time()
    loaderloss = 0.0
    for src_batch, tgt_batch in train_dataloader:
        src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
        src_mask, tgt_mask = generate_masks(src_batch, tgt_batch)
        src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)
        output = model(src_batch, tgt_batch, src_mask, tgt_mask)
        criterion = RMSELoss()
        loss = criterion(output, tgt_batch)
        loaderloss += loss.item()
    epoch_loss = loaderloss / len(train_dataloader) #avg loss over dataset
    step_time = time.time() - start_time
    return -epoch_loss  # PyGAD maximizes the fitness function, so we use the negative loss


def Xfrmr_GA(model, train_dataloader, num_generations=100, num_parents_mating=5, mutation_percent_genes=10):
    import pygad
    import pygad.torchga

    torch_ga = pygad.torchga.TorchGA(model=model, num_solutions=30)
    # Create an initial population
    initial_population = torch_ga.population_weights

    # Initialize loss history lists
    loss_history = []
    val_hist = []

    # Define a callback function to capture the loss history
    def callback_generation(ga_instance):
        global loss_history, val_hist
        solution, solution_fitness, _ = ga_instance.best_solution()
        loss_history.append(-solution_fitness)
        # Evaluate on the validation set
        criterion=RMSELoss()
        val_loss = getloss(model, criterion, val_dataloader)
        val_hist.append(val_loss)
        print(f"Generation {ga_instance.generations_completed}: Loss = {loss_history[-1]:.5f}, Validation Loss = {val_loss:.5f}")

    # Run the genetic algorithm
    ga_instance = pygad.GA(num_generations=3000,
                        num_parents_mating=5,
                        initial_population=initial_population,
                        fitness_func=fitness_func,
                        sol_per_pop=10,
                        parent_selection_type="sss",
                        keep_parents=2,
                        crossover_type="single_point",
                        mutation_type="random",
                        mutation_percent_genes=20,
                        on_generation=callback_generation)

    ga_instance.run()
    # Load the best solution into the model
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    model_weights_dict = pygad.torchga.model_weights_as_dict(model=model, weights_vector=solution)
    model.load_state_dict(model_weights_dict)
    torch.save(model.state_dict(), Xfrmr_path)
    return model, loss_history, val_hist

def getloss(model, criterion, dataloader):
    model.eval() # Set model to evaluation mode
    val_loss = 0
    with torch.no_grad():
        for src_batch, tgt_batch in dataloader:
            src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
            src_mask, tgt_mask = generate_masks(src_batch, tgt_batch)
            src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)
            output = model(src_batch, tgt_batch[:, :], src_mask, tgt_mask[:, :, :, :])
            loss = criterion(output, tgt_batch[:, :]) 
            val_loss += loss.item()
    val_loss /= len(dataloader)
    return val_loss

def Xfrmr_train(model, criterion, optimizer, scheduler, train_dataloader, val_dataloader, num_epochs=200, estop_patience=5, device='cpu'):
    model.to(device)
    model.train() # Set model to training mode
    loss_history = []
    val_hist = []
    early_stopping = EarlyStopping(patience=estop_patience, verbose=False)
    lr = optimizer.param_groups[0]['lr']

    for epoch in range(num_epochs):
        start_time = time.time()
        loaderloss = 0.0
        
        # Run optimizer loop for each batch in the training dataloader
        for src_batch, tgt_batch in train_dataloader:
            # print('batch shapes: ', src_batch.shape, tgt_batch.shape)
            src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
            src_mask, tgt_mask = generate_masks(src_batch, tgt_batch)
            # print('mask shapes: ', src_mask.shape, tgt_mask.shape)
            src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)
            optimizer.zero_grad()
            output = model(src_batch, tgt_batch, src_mask, tgt_mask)
            loss = criterion(output, tgt_batch)
            loss.backward()
            print(f'steploss: {loss.item(): .7f}')
            optimizer.step()
            loaderloss += loss.item()
            scheduler.step(loss.item())
            if optimizer.param_groups[0]['lr'] != lr:
                lr = optimizer.param_groups[0]['lr']
                print(f"LEARNING RATE: {lr}")
        epoch_loss = loaderloss / len(train_dataloader) #avg loss over dataset
        loss_history.append(epoch_loss)
        epoch_time = time.time() - start_time

        # Validation loss comparator. Consider multi-thread instead of 1/5 epochs
        if val_dataloader is not None:
            val_loss = getloss(model, criterion, val_dataloader)
            val_hist.append(val_loss)
            early_stopping(val_loss, model) # Check if model still improving
            if early_stopping.early_stop:
                print("Early stopping")
                torch.save(model.state_dict(), 'best_model.pt')
                break
            model.train() # return model to training mode
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.5f} - ValLoss: {val_loss:.5f} - StepTime[s]: {epoch_time:.4f}")

    
    print("Training Complete.")
    return model, loss_history, val_hist


if __name__ == "__main__":
    #---------- Data Initialization --------------------------------------------------------------------------
    import pandas as pd
    import os
    from matplotlib import pyplot as plt
    import json, joblib
    import datetime as dt
    from sklearn.preprocessing import StandardScaler
    import utils.data_process as prep
    from utils.optimization import RMSELoss, MAELoss, MAPELoss, WarmupScheduler
    from analysis import evaluate_ml as evaluate

    # Device configuration
    print(f'Using device: {device}')
    if device.type == 'cuda':
        print(f'Number of available GPUs: {torch.cuda.device_count()}')
        print(f'Current CUDA device: {torch.cuda.current_device()}')
        print(f'CUDA device name: {torch.cuda.get_device_name(device)}')
        print(f'CUDA device properties: {torch.cuda.get_device_properties(device)}')
    else:
        print("CUDA is not available. Using CPU.")

    training_path = r"C:\MLData\.." 
    data = prep.GrabData(training_path, db='training', weather_path=None, data_path=None, input_path=None)

    energy_inputs = ['QHeatPowerHEAT', 'OutletTemperature', 'DT', 'HeatEnergy_Cumulat', 'Heating_On', 'VolumeFlowHEAT'] #removed VolumeFlowHEAT
    parameter_inputs = ['year', 'age', 'Latitude', 'Longitude', 'floors', 'height', 'length', 'width']
    boundary_inputs = ['N_Gamma', 'SharedWalls','N_wallshare', 'E_wallshare', 'W_wallshare', 'S_wallshare']
    data_inputs = [input.lower() for input in energy_inputs + parameter_inputs + boundary_inputs]
    X = data[data_inputs] # Select the input columns
    # print(X.head())

    target_labels = [
                        'gValueWindows',
                        'ULambdaExWindows', 'ULambdaExWalls', 'ULambdaGrdFloor', 'ULambdaRoof', 
                    #   'LambdaExWalls',	'LambdaGrdFloor',	'LambdaRoof',	
                    #   'ThicknessExWalls',	'ThicknessGrdFloor',	'ThicknessRoof', 
                       'RadiatorSize', 
                        'AirInfiltration'
                      ] 
    target_columns = [col.lower() for col in target_labels]
    y = data[target_columns]
    # print(y.head())

    max_context = 240 # Maximum context length for the Transformer model. VERY IMPORTANT TO SET PROPERLY
    config = {
        "input_dim": len(data_inputs),  # Number of input features
        "output_dim": len(target_columns),  # Number of output features
        "emb_sz": 64,  # Embedded information dimensionality
        "num_heads": 8,  # Parallel attention heads. Must divide embedding size evenly
        "num_layers": 1,  # Number of encoder layers
        "d_ff": 128,  # Feed-forward layer dimensionality
        "nl_ff": 2,  # Number of feed-forward layers
        "dropout": 0.125,  # Dropout rate between layers
        "max_seq_length": int(max_context),  # Maximum context length
        'Xscaler_path': 'placeholder',
        'Yscaler_path': 'placeholder'
    }
    print('Input Dimension:', config["input_dim"], 'Output Dimension:', config["output_dim"])

    filter_params = {
        'filter_type': 'butter', # Filter type: 'filtfilt', 'butter' or 'biquadratic'
        'cutoff_freq': 10,  # Cutoff frequency in 1/days
        'order': 2}
    energy_inputs_lower = [input.lower() for input in energy_inputs]
    filter_cols = [i for i, input in enumerate(data_inputs) if input in energy_inputs_lower]

    periods = data['period'].values
    zero_indices = np.where(periods == 0)[0]
    X_list, X_list_og, y_list = prep.process_lists(X, y, zero_indices, max_context, filter_cols, filter_params) # Filter and clip data, arrange into lists of sequences
    max_context = max(df.shape[0] for df in X_list)
    print('Context Length: ', max_context)
    # prep.plot_data(X_list, energy_inputs, indices=[i for i, input in enumerate(data_inputs) if input in energy_inputs_lower], data2=X_list_og, title='Filtered Data')
    # plt.show()
    
    # from sklearn.decomposition import PCA
    # # Perform PCA
    # pca = PCA()
    # pca.fit(scaled_data)
    
    # Create Dataset and DataLoader
    train_data, val_data, train_target, val_target = train_test_split(X_list, y_list, test_size=0.05, random_state=42) # random_state=42

    # # Standardize the data for good practice training. Remember to denormalize with training params
    train_data, val_data, train_target, val_target, Xscaler, Yscaler = prep.scale_data(train_data, val_data, train_target, val_target)
    train_dataset = TimeSeriesDataset(train_data, train_target)
    val_dataset = TimeSeriesDataset(val_data, val_target)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, collate_fn=TimeSeriesDataset.collate_fn) # def batch size
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=256, collate_fn=TimeSeriesDataset.collate_fn)

    analysis = False
    model = Attentive_FFNN(config)
    params = count_parameters(model)
    print(f"Total number of learnable parameters in the model: {params}")
    model.to(device)
    
    # Path to the pretrained model
    Xfrmr_path = r'Results\models\AttendFFNN' + '_i' + str(config["input_dim"]) + '_o' + str(config["output_dim"]) + 'nl' + str(config["num_layers"]) + '_emb' + str(config["emb_sz"])+ '_df' + str(config["d_ff"]) 
    model_path = Xfrmr_path + '\model.pth'
    config_path = Xfrmr_path + "\config.json"
    Xscaler_path = Xfrmr_path + '\Xscaler.joblib'
    Yscaler_path = Xfrmr_path + '\Yscaler.joblib'
    config['scaler_path'] = Xscaler_path
    config['Yscaler_path'] = Yscaler_path

    # Load the pretrained model if parameters match
    if os.path.isfile(model_path):
        if model_params_match(model, model_path):
            print("Pretrained model found with matching Param counts..")
            from pytimedinput import timedInput
            userText, timedOut = timedInput(f"Overwrite Model? (y/n): ", timeout=25)
            if timedOut:
                print("Timed out when waiting for input. Defaulting to training new model.")
                new = 'y'
                proceed = 'y'
            else:
                new = userText.lower()
            if new.lower() == 'n':
                model.load_state_dict(torch.load(model_path))
            else:
                id = time.time()
                print("Creating new model...")
        else:
            print("Different Param count than pretrained model. Creating new model...")
            new = 'y'
    else:
        print("Pretrained model not found. Creating new model...")
        new = 'y'
    if not os.path.exists(Xfrmr_path):
        print("Creating new directory...")
        os.makedirs(Xfrmr_path)


    ######## Training Loop #########
    num_epochs = 75
    estop_patience= 6
    criterion = nn.MSELoss() # Criterion for loss calculation. Options: nn.MSELoss, RMSELoss, MAELoss, MAPELoss
    optimizer = optim.Adam(model.parameters(), lr=0.000009, betas=(0.99, 0.99), eps=1e-9)
    warmup_steps = 100
    total_steps = len(train_dataloader) * num_epochs
    base_lr = 0.00001
    peak_lr = 0.0009
    scheduler = WarmupScheduler(optimizer, warmup_steps, total_steps, base_lr, peak_lr)
    Trainer = 'Adam' # 'GA' or 'Adam' Note: GA training takes forever, and produces similar convergence. Training with Adam is highly recommended.
    
    if new != 'y':
        userText, timedOut = timedInput(f"Proceed with Training?(y/n): ", timeout=15)
        if timedOut:
            print("Timed out when waiting for input.")
            print(f"Defaulting to proceed with training.")
            proceed = 'y'  # Default to 'y' if timed out
        else:
            proceed = userText.lower()
    else: proceed = 'y'

    if proceed.lower() == 'y':
        print("Training Model...")
        if Trainer == 'GA': # Genetic Algorithm Training
            model, loss_history, val_hist = Xfrmr_GA(model, train_dataloader, num_generations=100, num_parents_mating=5, mutation_percent_genes=10)
        else: # Conventional Adam Training
            model, loss_history, val_hist = Xfrmr_train(model, criterion, optimizer, scheduler,
                                                        train_dataloader, val_dataloader, 
                                                        num_epochs=num_epochs, estop_patience=estop_patience, device=device)
        val_loss = getloss(model, criterion, val_dataloader)
        print(f"Validation Loss: {val_loss}")

        fig1 = evaluate.eval_loss(loss_history, val_hist, val_loss, criterion)

        print("Saving model... Path: " + model_path)
        if os.path.isfile(model_path):
            os.remove(model_path)
        torch.save(model.state_dict(), model_path)
        model.eval() # Set model to evaluation mode for analysis

        # Save the config dictionary to a JSON file
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

        # Save the scaler object to a joblib file
        joblib.dump(Xscaler, Xscaler_path)
        joblib.dump(Yscaler, Yscaler_path)

    else:
        print("Skipping Training.. using pretrained model.")
        model = Attentive_FFNN(config)   # Define model architecture
        model.load_state_dict(torch.load(model_path))     # Load the state dictionary into the model
        model.to(device)    # Move the model to the appropriate device
        with open(config_path, 'r') as f:
            config = json.load(f)
        model.eval() # Set model to evaluation mode: no dropout, gradient


    #---------- Model Evaluation --------------------------------------------------------------------------
    print("Model Evaluation...")
    architecture_analysis = False
    # Function to get a single sample from the DataLoader batch
    def get_sample_from_dataloader(dataloader, device):
        for src_batch, tgt_batch in dataloader:
            # Select a single sample from the batch
            index = np.random.randint(0, src_batch.size(0))  # Random index
            src_sample = src_batch[index].unsqueeze(0).to(device)  # Add batch dimension
            tgt_sample = tgt_batch[index].unsqueeze(0).to(device)  # Add batch dimension
            
            return src_sample, tgt_sample

    src_sample, tgt_sample = get_sample_from_dataloader(val_dataloader, device)
    with torch.no_grad():
        max_seq_length = src_sample.size(1)
        predictions = predict(model, src_sample, max_seq_length, device=device)
    
    # Extract predictions and targets for the selected sample
    predictions = predictions.squeeze(0)  # Remove batch dimension
    targets = tgt_sample.squeeze(0).cpu().numpy()  # Remove batch dimension

    # Denormalize the predictions and targets
    predictions = prep.inverse_transform(predictions, Yscaler)
    targets = prep.inverse_transform(targets, Yscaler)

    # Plot predictions vs targets for each dimension
    fig2 = evaluate.predictions_vs_targets(predictions, targets, target_columns)
    architecture_analysis = False

    # Assuming varnames is a list of names for each dimension
    varnames = target_columns  # or any other list of variable names you have

    # Generate predictions for the validation set
    val_outputs = []
    val_targets = []

    with torch.no_grad():
        for src_batch, tgt_batch in val_dataloader:  # We need both src_batch and tgt_batch for plotting
            src_batch = src_batch.to(device)
            max_seq_length = src_batch.size(1) 
            output = predict(model, src_batch, max_seq_length, device=device)
            val_outputs.append(output)
            val_targets.append(tgt_batch.cpu().numpy())

    print("length of outputs:", len(val_outputs))
    val_outputs = np.concatenate(val_outputs, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)
    print('val_outputs shape:', val_outputs.shape)

    val_outputs = prep.inverse_transform(val_outputs, Yscaler)
    val_targets = prep.inverse_transform(val_targets, Yscaler)

    print(f"Validation Predictions shape: {val_outputs.shape}")
    print(f"Validation Targets shape: {val_targets.shape}")

    # Plot scatter charts for each dimension
    fig3 = evaluate.plot_scatter_predictions_vs_targets(val_outputs, val_targets, varnames)
    
    plt.show()