from __future__ import print_function
import torch, os, numpy as np, math, sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.distributions.categorical import Categorical
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.parameter import Parameter
from torch.nn.modules.loss import CrossEntropyLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


##########################################################################
##########################################################################
def main():

    # Settings
    batch_size = 16
    data_dir = './data'
    max_num_epochs = 100

    # Data retrieval
    train_loader, test_loader = load_mnist(data_dir,batch_size)

    # Setup
    input_shape = (28,28)
    output_shape = (1,10)
    gs = Graph_sampler(input_shape,
                       output_shape,
                       batch_size = batch_size,
                       M = 2,
                       num_sample = 25
                      ).to(device).type(torch.float)

    # Run training optimization
    # params = gs.state_dict()
    # for param in params:
    #     print(param,params[param])
    # sys.exit(0)

    optimizer_classification = optim.Adam(gs.parameters(), lr=0.00003)
    optimizer_reinforce = optim.Adam(gs.parameters(), lr=0.00003)
    ce_loss = CrossEntropyLoss()
    for epoch in range(max_num_epochs):
        loss_arr = []
        reward_arr = []
        ce_loss_arr = []
        for batch_idx, (data, label) in enumerate(train_loader):
            label = label.to(device).type(torch.cuda.LongTensor)
            pred_output = gs.forward(data.to(device).type(torch.float))

            classification_loss = ce_loss(pred_output, label)

            values, indices = torch.max(pred_output, 1)
            # print('IL',indices, label)
            reward = torch.sum(indices == label)
            # print(reward,len(label))

            # Note that this is equivalent to what used to be called multinomial
            m = Categorical(pred_output)
            action = m.sample()
            loss = -m.log_prob(action) * reward.type(torch.cuda.FloatTensor)
            loss = loss.sum() / batch_size # make invariant to changing batch_size

            optimizer_classification.zero_grad()
            classification_loss.backward()
            optimizer_classification.step()

            loss_arr.append(loss)
            reward_arr.append(reward)
            ce_loss_arr.append(classification_loss)

            if(batch_idx % 5 == True):
                mu_r = np.mean(torch.stack(reward_arr).cpu().detach().numpy())
                mu_ce = np.mean(torch.stack(ce_loss_arr).cpu().detach().numpy())
                print('-'*30)
                print("Epoch: " + str(epoch) + " batch: " + str(batch_idx) +
                      " Reward Avg: ",  mu_r, "CERR: " + str(mu_ce))
                print('Params (sampler)\n', gs.m_g_params[0])
                # print('Params (non-sampler)\n', gs.)
            #break
        print("Epoch" +  str(epoch) + "loss: ",
            np.mean(torch.stack(reward_arr).cpu().detach().numpy())
        )

##########################################################################
##########################################################################

def load_mnist(data_dir,batch_size):
    if not os.path.exists(data_dir): os.mkdir(data_dir)
    trans = transforms.Compose([ transforms.ToTensor(),
                                # MNIST specific normalization
                                 transforms.Normalize((0.1307,), (0.3081,)) ])
    train_set =  datasets.MNIST(data_dir, train=True, download=True, transform=trans)
    test_set  = datasets.MNIST(data_dir, train=False, transform=trans, download=True)
    train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
    test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)
    print('Loaded MNIST')
    return train_loader, test_loader

##########################################################################

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        input = input.cuda()
        adj = adj.cuda()
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

##########################################################################

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid).to(device).type(torch.float)
        self.gc2 = GraphConvolution(nhid, nclass).to(device).type(torch.float)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x)

##########################################################################

class Graph_sampler(nn.Module):
    def __init__(self, input_shape, output_size, output=10, num_features=1,
                 hidden_graph=100, M=10, num_sample=20, dropout=0.2,
                 batch_size=100):
        super(Graph_sampler, self).__init__()

        self.batch_size = batch_size
        self.num_sample = num_sample
        self.gcn = GCN(num_features, hidden_graph, output, dropout).to(device).type(torch.float)
        self.max_pooling = nn.MaxPool1d(20)
        self.fcf = nn.Linear(output*num_sample,output)
        self.output_gcn = output
        # Sampling parameters
        # There are M bivariate gaussian: Each bivariate gaussian has 5
        # parameters + vector M which contains the mixture weight
        self.m_g_params = Parameter(torch.randn(6*M).to(device))

    def forward(self, x):
        self.params = torch.split(self.m_g_params, 6, 1)
        self.params_mixture = torch.stack(self.params)
        self.pi,self.mu_x,self.mu_y,self.sigma_x,self.sigma_y,self.rho_xy = torch.split(self.params_mixture,1,2)
        self.pi = self.pi.squeeze().view(self.batch_size,-1)
        self.pi = F.softmax(self.pi,dim=-1)
        self.mu_x = self.mu_x.squeeze().view(self.batch_size,-1)
        self.mu_y = self.mu_y.squeeze().view(self.batch_size,-1)
        self.sigma_x =  self.sigma_x.squeeze().view(self.batch_size,-1)
        self.sigma_y =  self.sigma_y.squeeze().view(self.batch_size,-1)
        self.rho_xy =  self.rho_xy.squeeze().view(self.batch_size,-1)
        samples = []
        for i in range(self.num_sample):
            loc = self.sample_points()
            pixel_value = self.get_pixel_value(x,loc) #currently not doing bilinear interpolation
            samples.append(pixel_value)
            #break
        output = torch.stack(samples)
        # Vector of node features
        feature_matrix = output.view(self.batch_size,self.num_sample, -1)
        # Adjacency weight matrix
        adject_matrix = torch.ones(self.num_sample,self.num_sample)
        output = self.gcn(feature_matrix,adject_matrix)
        output = output.view(self.batch_size,  self.output_gcn*self.num_sample)
        output = F.softmax(self.fcf(output), dim=-1)
        #output = self.max_pooling(output)
        return output.squeeze(-1)

    def sample_points(self):
        #can do reparametrization trick: https://arxiv.org/abs/1607.05690
        m = Categorical(self.pi)
        pi_idx = m.sample().to(device)
        pi_idx = pi_idx.view(-1,1)
        mu_x = torch.gather(self.mu_x,1,pi_idx)
        mu_y = torch.gather(self.mu_y,1,pi_idx)
        sigma_x = torch.exp(torch.gather(self.sigma_x,1,pi_idx))
        sigma_y = torch.exp(torch.gather(self.sigma_y,1,pi_idx))
        rho_xy = torch.tanh(torch.gather(self.rho_xy,1,pi_idx))
        loc =  self.sample_bivariate_normal(mu_x,mu_y,sigma_x,sigma_y,rho_xy, greedy=False)
        return loc

    def sample_bivariate_normal(self, mu_x,mu_y,sigma_x,sigma_y,rho_xy, greedy=False):
        if greedy: return mu_x, mu_y
        mean = torch.stack((mu_x, mu_y), dim=-1).squeeze()
        cov_axis0 = torch.stack([sigma_x * sigma_x, rho_xy * sigma_x * sigma_y], dim=-1).squeeze()
        cov_axis1 = torch.stack([rho_xy * sigma_x * sigma_y, sigma_y * sigma_y], dim=-1).squeeze()
        cov = torch.stack((cov_axis0,cov_axis1), dim=-1)
        m = MultivariateNormal(mean, cov)
        x = m.sample()
        return F.tanh(x) #Normalize it to -1 to 1 to get pixel value

    def get_pixel_value(self,x,loc):
        B, C, H, W = x.shape
        denorm_loc = self.denormalize(H, loc) #height and width are same
        #denorm_loc =torch.split(denorm_loc.unsqueeze(1),1,2)
        pixel_features = []
        for i,image in enumerate(x):
            pixel_value = image[:,denorm_loc[i][0],denorm_loc[i][1]]
            pixel_features.append(pixel_value)
        pixel_features = torch.stack(pixel_features)
        return pixel_features

    def denormalize(self, T, coords):
        return (0.5 * ((coords + 1.0) * T) - 0.1).long()


##########################################################################
##########################################################################
if __name__ == '__main__': main()
##########################################################################
##########################################################################

# conda create -n attn python=3.6 scipy pytorch torchvision
# source activate attn