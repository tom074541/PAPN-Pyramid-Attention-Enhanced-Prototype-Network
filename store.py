import torch
import torch.nn as nn
import math
import torch.nn.functional as F
#VGG16
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3,padding=1),  # Conv1_1
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),          # Conv1_2
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),                # MaxPool1

            nn.Conv1d(64, 128, kernel_size=3, padding=1),         # Conv2_1
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),        # Conv2_2
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),                # MaxPool2

            nn.Conv1d(128, 256, kernel_size=3, padding=1),        # Conv3_1
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),        # Conv3_2
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),        # Conv3_3
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),                # MaxPool3

            nn.Conv1d(256, 512, kernel_size=3, padding=1),        # Conv4_1
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),        # Conv4_2
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),        # Conv4_3
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),                # MaxPool4

            nn.Conv1d(512, 512, kernel_size=3, padding=1),        # Conv5_1
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),        # Conv5_2
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),        # Conv5_3
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)                 # MaxPool5
        )
        self.avgpool = nn.AdaptiveAvgPool1d(7)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

#Simple CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  
            nn.Conv1d(in_channels=1, 
                      out_channels=16,  
                      kernel_size=5,  
                      stride=1,  
                      padding=2  
                      ),  
            nn.ReLU(),  
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(nn.Conv1d(16, 32, 5, 1, 2),
                                   nn.ReLU(),
                                   nn.MaxPool1d(2))

    def forward(self, x):
        x = self.conv1(x)  
        x = self.conv2(x)  
        x = x.view(x.size(0), -1)  
        return x
    
#Resnet18
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out


class Resnet18(nn.Module):

    def __init__(self):
        super(Resnet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(1, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
#LSTM
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        x=x.permute(0,2,1) #input:(batch,channel,sequence length) LSTM need:(batch,sequence length,input size(channel))
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        return out
    
class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x
        return x
    
#LSTM with multi-head attention

class LSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads):
        super(LSTMAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))

        attn_output, attn_output_weights = self.multihead_attn(out, out, out)

        final_output = attn_output[:, -1, :]

        return final_output

#LSTM with multi-head attention and Sinusoidal position encoding

class LSTMAttentionEncoding(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads):
        super(LSTMAttentionEncoding, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        self.position_encodings = nn.Parameter(self.init_sinusoidal_pos_enc(hidden_size), requires_grad=False)
    
    def init_sinusoidal_pos_enc(self, hidden_size, max_len=500):
        pos_enc = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))

        seq_len = x.size(1)
        out =out+self.position_encodings[:seq_len, :]

        attn_output, attn_weights = self.multihead_attn(out, out, out)

        final_output = attn_output[:, -1, :]
        
        return final_output

#Simple CNN with attention and positional encoding
class CNNAttention(nn.Module):
    def __init__(self,d_model, num_heads):
        super(CNNAttention, self).__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.conv1 = nn.Sequential(  
            nn.Conv1d(in_channels=1, 
                      out_channels=16,  
                      kernel_size=5,  
                      stride=1,  
                      padding=2  
                      ), 
            nn.ReLU(),  
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(nn.Conv1d(16, 32, 5, 1, 2),
                                   nn.ReLU(),
                                   nn.MaxPool1d(2))
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        self.position_encodings = nn.Parameter(self.init_sinusoidal_pos_enc(d_model), requires_grad=False)
    def init_sinusoidal_pos_enc(self, d_model, max_len=5000):
        pos_enc = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc

    def forward(self, x):
        x = self.conv1(x)  
        x = self.conv2(x)  
        x=x.permute(0,2,1)
        seq_len = x.size(1)
        x =x+self.position_encodings[:seq_len, :]
        attn_output, attn_weights = self.multihead_attn(x, x, x)
        #attn_output = attn_output.reshape(attn_output.size(0), -1)
        attn_output= torch.mean(attn_output, dim=1)
        return attn_output
    
#CNN+multi-head attention+time attention
class CNNTimeAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(CNNTimeAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        
        # Sinusoidal positional encoding initialization
        self.position_encodings = nn.Parameter(self.init_sinusoidal_pos_enc(d_model), requires_grad=False)

        # Time Attention weights
        self.time_attn_weights = nn.Parameter(torch.rand(d_model), requires_grad=True)

    def init_sinusoidal_pos_enc(self, d_model, max_len=5000):
        pos_enc = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1) 
        seq_len = x.size(1)

        # Apply positional encodings
        x = x + self.position_encodings[:seq_len, :]

        # Apply initial Multihead Attention
        attn_output, attn_weights = self.multihead_attn(x, x, x)
        
        # Apply Time Attention
        time_attn_output = attn_output * self.time_attn_weights

        # Average the outputs
        final_output = torch.mean(time_attn_output, dim=1)

        return final_output

#Pyramid attention network
class PyramidAttention(nn.Module):
    def __init__(self, d_model, scales=[1, 2, 4]):
        super(PyramidAttention, self).__init__()
        self.scales = scales
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, 8, batch_first=True) for _ in scales
        ])

    def forward(self, x):
        outs = []
        original_shape = x.shape
        target_size = original_shape[2]  

        for i, scale in enumerate(self.scales):
            if scale > 1:
                x_pooled = F.max_pool1d(x.transpose(1, 2), kernel_size=scale, stride=scale).transpose(1, 2)
            else:
                x_pooled = x

            attn_output, _ = self.attention_layers[i](x_pooled, x_pooled, x_pooled)
            attn_output = F.interpolate(attn_output.transpose(1, 2), size=target_size).transpose(1, 2)
            outs.append(attn_output)
        
        out = torch.mean(torch.stack(outs), dim=0)
        return out

class CNNPAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(CNNPAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.conv1 = nn.Sequential(  
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),  
            nn.ReLU(),  
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(nn.Conv1d(16, 32, 5, 1, 2),
                                   nn.ReLU(),
                                   nn.MaxPool1d(2))
        self.pyramid_attn = PyramidAttention(d_model, scales=[1, 2, 4])
        self.position_encodings = nn.Parameter(self.init_sinusoidal_pos_enc(d_model), requires_grad=False)

    def init_sinusoidal_pos_enc(self, d_model, max_len=5000):
        pos_enc = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc

    def forward(self, x):
        x = self.conv1(x)  
        x = self.conv2(x)  
        x = x.permute(0, 2, 1)
        seq_len = x.size(1)
        x = x + self.position_encodings[:seq_len, :]
        attn_output = self.pyramid_attn(x)
        final_output = torch.mean(attn_output, dim=1)
        return final_output

#CNN+Attention+gate
class CNNAttentionGate(nn.Module):
    def __init__(self, d_model, num_heads):
        super(CNNAttentionGate, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Convolutional layers
        self.conv1 = nn.Sequential(  
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),  
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Multihead attention
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        
        # Positional encodings
        self.position_encodings = nn.Parameter(self.init_sinusoidal_pos_enc(d_model), requires_grad=False)
        
        # Gate mechanism
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.Sigmoid()
        )

    def init_sinusoidal_pos_enc(self, d_model, max_len=5000):
        pos_enc = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc

    def forward(self, x):
        # CNN processing
        cnn_out = self.conv1(x)
        cnn_out = self.conv2(cnn_out)
        cnn_out = cnn_out.permute(0, 2, 1)  # Adjust shape for attention
        
        # Attention processing
        seq_len = x.size(1)
        x = x.permute(0, 2, 1)
        x = x + self.position_encodings[:seq_len, :]
        attn_out, attn_weights = self.multihead_attn(x, x, x)
        
        # Prepare for gating
        cnn_out = torch.mean(cnn_out, dim=1)
        attn_out = torch.mean(attn_out, dim=1)
        
        # Concatenate outputs
        combined = torch.cat((cnn_out, attn_out), dim=1)
        
        # Apply gate
        gate_values = self.gate(combined)
        gated_output = combined * gate_values
        
        return gated_output


#Resnet18+Multi-head attention
class PyramidAttention(nn.Module):
    def __init__(self, d_model, scales=[1, 2, 4]):
        super(PyramidAttention, self).__init__()
        self.scales = scales
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, 8, batch_first=True) for _ in scales
        ])

    def forward(self, x):
        outs = []
        original_shape = x.shape
        target_size = original_shape[2]  

        for i, scale in enumerate(self.scales):
            if scale > 1:
                x_pooled = F.max_pool1d(x.transpose(1, 2), kernel_size=scale, stride=scale).transpose(1, 2)
            else:
                x_pooled = x

            attn_output, _ = self.attention_layers[i](x_pooled, x_pooled, x_pooled)
            attn_output = F.interpolate(attn_output.transpose(1, 2), size=target_size).transpose(1, 2)
            outs.append(attn_output)
        
        out = torch.mean(torch.stack(outs), dim=0)
        return out
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out


class Resnet18Attention(nn.Module):

    def __init__(self, d_model, num_heads):
        super(Resnet18Attention, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(1, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.adapt_conv = nn.Linear(512, 32)
        # Multihead attention
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        #self.pyramid_attn = PyramidAttention(d_model, scales=[1, 2, 4])

        # Positional encodings
        self.position_encodings = nn.Parameter(self.init_sinusoidal_pos_enc(d_model), requires_grad=False)
        
        # Gate mechanism
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.Sigmoid()
        )

    def init_sinusoidal_pos_enc(self, d_model, max_len=5000):
        pos_enc = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        #Resnet processing
        Resout = self.conv1(x)
        Resout = self.bn1(Resout)
        Resout = self.relu(Resout)
        Resout = self.maxpool(Resout)
        Resout = self.layer1(Resout)
        Resout = self.layer2(Resout)
        Resout = self.layer3(Resout)
        Resout = self.layer4(Resout)
        Resout = self.avgpool(Resout)
        Resout=Resout.permute(0,2,1)
        Resout=self.adapt_conv(Resout)
        # Attention processing
        seq_len = Resout.size(1)
        Resout = Resout.permute(0, 2, 1)
        Resout = Resout + self.position_encodings[:seq_len, :]
        #attn_output = self.pyramid_attn(Resout)
        attn_output, attn_weights = self.multihead_attn(Resout, Resout, Resout)
        attn_output= torch.mean(attn_output, dim=1)
        return attn_output
        
#Resnet18+Pyramid attention network
class PyramidAttention(nn.Module):
    def __init__(self, d_model, scales=[1, 2, 4]):
        super(PyramidAttention, self).__init__()
        self.scales = scales
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, 8, batch_first=True) for _ in scales
        ])

    def forward(self, x):
        outs = []
        original_shape = x.shape
        target_size = original_shape[2]  

        for i, scale in enumerate(self.scales):
            if scale > 1:
                x_pooled = F.max_pool1d(x.transpose(1, 2), kernel_size=scale, stride=scale).transpose(1, 2)
            else:
                x_pooled = x

            attn_output, _ = self.attention_layers[i](x_pooled, x_pooled, x_pooled)
            attn_output = F.interpolate(attn_output.transpose(1, 2), size=target_size).transpose(1, 2)
            outs.append(attn_output)
        
        out = torch.mean(torch.stack(outs), dim=0)
        return out
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out


class Resnet18PAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super(Resnet18PAttention, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(1, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.adapt_conv = nn.Linear(512, 32)
        self.pyramid_attn = PyramidAttention(d_model, scales=[1, 2, 4])

        # Positional encodings
        self.position_encodings = nn.Parameter(self.init_sinusoidal_pos_enc(d_model), requires_grad=False)
        
        # Gate mechanism
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.Sigmoid()
        )

    def init_sinusoidal_pos_enc(self, d_model, max_len=5000):
        pos_enc = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        #Resnet processing
        Resout = self.conv1(x)
        Resout = self.bn1(Resout)
        Resout = self.relu(Resout)
        Resout = self.maxpool(Resout)
        Resout = self.layer1(Resout)
        Resout = self.layer2(Resout)
        Resout = self.layer3(Resout)
        Resout = self.layer4(Resout)
        Resout = self.avgpool(Resout)
        Resout=Resout.permute(0,2,1)
        Resout=self.adapt_conv(Resout)
        # Attention processing
        seq_len = Resout.size(1)
        Resout = Resout.permute(0, 2, 1)
        Resout = Resout + self.position_encodings[:seq_len, :]
        attn_output = self.pyramid_attn(Resout)
        attn_output= torch.mean(attn_output, dim=1)
        return attn_output