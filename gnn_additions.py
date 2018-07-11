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
from torch.autograd import Variable



allow_gpu = True
if allow_gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device('cpu')

##########################################################################
##########################################################################
def main():

    #----------------------------------------------------------------#
    # Parameters
    batch_size = 4
    n_samples_per_img = 5
    reinforce_lr = 0.025
    classification_lr = 0.000125
    graph_sig_size = 5

    # Settings
    data_dir = './data'
    max_num_epochs = 100
    n_mixture_components = 2
    reward_type = 2 # 1 = negative CE_loss, 2 = 0-1 correctness
    #----------------------------------------------------------------#


    # Data retrieval
    train_loader, test_loader = load_mnist(data_dir,batch_size)

    # Setup
    input_shape = (28,28)
    output_shape = (1,10)
    gs = Graph_sampler(input_shape,
                       output_shape,
                       batch_size = batch_size,
                       M = n_mixture_components,
                       num_sample = n_samples_per_img,
                       sig_size = graph_sig_size
                      ).to(device).type(torch.float)

    if reward_type == 1:
        rewardf = lambda i,l,cl: -cl
    elif reward_type == 2:
        rewardf = lambda i,l,cl: (i == l).type(torch.cuda.FloatTensor)

    targ_param_name = 'm_g_params'
    c_params = [ p[1] for p in gs.named_parameters()
                 if not p[0] == targ_param_name ]

    optimizer_classification = optim.Adam(c_params, lr=classification_lr)
    optimizer_reinforce = optim.Adam([gs.m_g_params], lr=reinforce_lr)
    ce_loss = CrossEntropyLoss(reduce=False)
    for epoch in range(max_num_epochs):
        loss_arr = []
        reward_arr = []
        ce_loss_arr = []
        for batch_idx, (data, label) in enumerate(train_loader):
            ### Retrieve true label ###
            label = label.to(device).type(torch.cuda.LongTensor)
            ### Run network forward pass on batch ###
            pred_output, logprob = gs.forward(data.to(device).type(torch.float))
            ### Get classification loss for all differentiable parts of the network ###
            classification_losses = ce_loss(pred_output, label)
            classification_loss = classification_losses.mean() # Note reduce=False
            ### Get REINFORCE loss to train the sampler ###
            values, indices = torch.max(pred_output, 1)
            # Actual reward value
            rewards = rewardf(indices, label, classification_losses).detach()
            #
            print('\tR=',rewards.sum())
            #
            rloss = (-logprob * rewards / batch_size).sum()

            ### Backprop training for differentiable network components ###
            optimizer_classification.zero_grad()
            classification_loss.backward()
            optimizer_classification.step()

            ### REINFORCE gradient update for the sampler ###
            optimizer_reinforce.zero_grad()
            rloss.backward()
            optimizer_reinforce.step()

            ### Track progress ###
            loss_arr.append(rloss)
            reward_arr.append(rewards)
            ce_loss_arr.append(classification_loss)

            if(batch_idx % 5 == True):
                mu_r = np.mean(torch.stack(reward_arr).cpu().detach().numpy())
                mu_ce = np.mean(torch.stack(ce_loss_arr).cpu().detach().numpy())
                print('-'*30)
                print("Epoch: " + str(epoch) + " batch: " + str(batch_idx) +
                      " Reward Avg: ",  mu_r, "CERR: " + str(mu_ce))
                print('Params (sampler)\n', gs.m_g_params)
                # print('Params (non-sampler)\n', c_params)
            #break
        print("Epoch" +  str(epoch) + "rewards: ",
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

class Graph_sampler(nn.Module):
    def __init__(self, input_shape, output_size, output=10, num_features=3,
                 hidden_graph=100, M=10, num_sample=20, dropout=0.2,
                 batch_size=100, sig_size=100):
        super(Graph_sampler, self).__init__()

        self.batch_size = batch_size
        self.num_sample = num_sample
        self.sig_size = sig_size
        # self, nfeat, nhid, sig_size, n_samples, dropout
        self.gcn = GCNsig(
                    num_features, hidden_graph, sig_size,
                    num_sample, dropout
                    ).to(device).type(torch.float)
        self.max_pooling = nn.MaxPool1d(20)
        self.fcf = nn.Linear(sig_size, output)
        self.output_gcn = output
        self.M = M
        # Sampling parameters
        # There are M bivariate gaussian: Each bivariate gaussian has 5
        # parameters + vector M which contains the mixture weight
        self.m_g_params = Parameter(torch.randn(1, 6*self.M).to(device))
        # Parameters for similarity (inverse distance) weighted adjacency
        self.alpha = 1.0
        self.beta = 0.5

    def forward(self, x):
        self.sparams = torch.split(self.m_g_params.expand(
                                        self.num_sample,6*self.M), 6, 1)
        self.params_mixture = torch.stack(self.sparams)
        (self.pi, self.mu_x, self.mu_y,
         self.sigma_x, self.sigma_y,
         self.rho_xy) = torch.split(self.params_mixture,1,2)
        self.pi = self.pi.squeeze().view(self.num_sample,-1)
        self.pi = F.softmax(self.pi, dim=-1)
        self.mu_x = self.mu_x.squeeze().view(self.num_sample,-1)
        self.mu_y = self.mu_y.squeeze().view(self.num_sample,-1)
        self.sigma_x =  self.sigma_x.squeeze().view(self.num_sample,-1)
        self.sigma_y =  self.sigma_y.squeeze().view(self.num_sample,-1)
        self.rho_xy =  self.rho_xy.squeeze().view(self.num_sample,-1)
        samples, adj_mats = [], []
        full_log_prob = Variable(torch.zeros(self.batch_size), requires_grad=True).to(device)
        for i in range(self.batch_size):
            loc, logprob, adj_mat = self.sample_points()
            pixel_value = self.get_pixel_value_single_img(x,loc,i)
            # print("xs",x.size())
            # print("locs",loc.size())
            # print("logprobs",logprob.size())
            # print("adjmat",adj_mat.size())
            # print("pval",pixel_value.size()) # n_samples x 1
            samples.append( torch.cat([pixel_value, loc], 1) )
            # full_log_prob = full_log_prob + logprob <-- across samples
            full_log_prob[i] = logprob.sum() # sum logprobs over samples to get log prob for a single batch member
            adj_mats.append(adj_mat)
            #break
        output = torch.stack(samples) # TODO does this stack and view (below) do the right thing?
        sim_mat = torch.stack(adj_mats)
        # print(sim_mat.size()) # batch_size x n_samples x n_samples
        # print(output.size()) # batch_size x n_samples x n_features
        # Vector of node features
        feature_matrix = output.view(self.batch_size, self.num_sample, -1)
        # Adjacency weight matrix
        #adject_matrix = torch.ones(self.num_sample, self.num_sample)
        adject_matrix = sim_mat
        output = self.gcn.forward_sig(feature_matrix, adject_matrix)
        output = output.view(self.batch_size,  self.sig_size)
        output = F.softmax(self.fcf(output), dim=-1)
        #output = self.max_pooling(output)
        return output.squeeze(-1), full_log_prob

    def sample_points(self):
        # Can do reparametrization trick: https://arxiv.org/abs/1607.05690
        m = Categorical(self.pi)
        pi_idx = m.sample().to(device)
        pi_idx = pi_idx.view(-1,1)
        mu_x = torch.gather(self.mu_x,1,pi_idx)
        mu_y = torch.gather(self.mu_y,1,pi_idx)
        sigma_x = torch.exp(torch.gather(self.sigma_x,1,pi_idx))
        sigma_y = torch.exp(torch.gather(self.sigma_y,1,pi_idx))
        rho_xy = torch.tanh(torch.gather(self.rho_xy,1,pi_idx))
        loc, logprob =  self.sample_bivariate_normal(mu_x,mu_y,sigma_x,sigma_y,rho_xy, greedy=False)

        # print("locs",loc.size())

        # Compute distance-weighted adjacency matrix
        # loc is n_samples x 2
        P_tilde = loc.expand(self.num_sample, self.num_sample, 2)
        D_unnorm_sq = ( P_tilde - P_tilde.transpose(0,1) ).pow(2).sum(dim=2)
        D_sim_sq = 1.0 / (self.alpha + self.beta * D_unnorm_sq)
        return loc, logprob, D_sim_sq

    def sample_bivariate_normal(self, mu_x,mu_y,sigma_x,sigma_y,rho_xy, greedy=False):
        if greedy: return mu_x, mu_y
        mean = torch.stack((mu_x, mu_y), dim=-1).squeeze()
        cov_axis0 = torch.stack([sigma_x * sigma_x, rho_xy * sigma_x * sigma_y], dim=-1).squeeze()
        cov_axis1 = torch.stack([rho_xy * sigma_x * sigma_y, sigma_y * sigma_y], dim=-1).squeeze()
        cov = torch.stack((cov_axis0,cov_axis1), dim=-1)
        m = MultivariateNormal(mean, cov)
        x = m.sample()
        logprob = m.log_prob(x)
        return F.tanh(x), logprob # Normalize it to -1 to 1 to get pixel value

    def get_pixel_value_single_img(self,x,locs,ind):
        B, C, H, W = x.shape
        denorm_locs = self.denormalize(H, locs) #height and width are same
        #denorm_loc =torch.split(denorm_loc.unsqueeze(1),1,2)
        # print('denorm_locs', denorm_locs.size())
        pixel_features = []
        targ_img = x[ind,:,:,:]
        for i,loc in enumerate(denorm_locs):
            pixel_value = targ_img[:,loc[0],loc[1]]
            pixel_features.append(pixel_value)
        pixel_features = torch.stack(pixel_features)
        # print('pixfeats', pixel_features.size())
        return pixel_features

    def get_pixel_value_across_images(self,x,loc):
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
        # input = input.to(device).type(torch.cuda.FloatTensor) #input.cuda()
        # adj = adj.to(device).type(torch.cuda.FloatTensor) #adj.cuda()
        adj = adj.cuda()
        input = input.cuda()
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

##########################################################################

class GCNsig(nn.Module):
    def __init__(self, nfeat, nhid, sig_size, n_samples, dropout):
        super(GCNsig, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid).to(device).type(torch.float)
        self.gc2 = GraphConvolution(nhid, nhid).to(device).type(torch.float)
        self.collapser1 = Parameter(torch.Tensor(sig_size, n_samples))
        self.collapser2 = Parameter(torch.Tensor(sig_size, n_samples))
        self.dropout = dropout
        self.softmax = nn.Softmax()

    def forward_sig(self, x, adj):
        # print('adj', adj.size())
        # print('x',x.size())

        # Run input graph (node feat & adj matrices)
        # Note: x in batch_size x n_samples x node_feats
        #       adj in n_samples x n_samples # <------- TODO by batch-size?!
        #       g_out in batch_size x n_samples x n_hidden_feats
        g_out1 = F.relu(self.gc1(x, adj))
        g_out1 = F.dropout(g_out1, self.dropout, training=self.training)
        g_out2 = F.relu(self.gc2(g_out1, adj))
        g_out2 = F.dropout(g_out2, self.dropout, training=self.training)

        # print('g_out1', g_out1.size())
        # print('g_out2', g_out2.size())

        # print('C1',self.collapser1)
        # print('g_out2', g_out2)

        # Collapse to a signature over the graph (linear transform)
        # Now: g_out in batch_size x n_samples x n_hidden_feats
        g_out1 = self.softmax( torch.matmul(self.collapser1, g_out1) )
        g_out2 = self.softmax( torch.matmul(self.collapser2, g_out2) )

        # print('g_out1c', g_out1.size())
        # print('g_out2c', g_out2.size())

        # Collapsed signature
        # Finally: sig in batch_size x sig_size
        sig = (g_out1 + g_out2).sum(dim=2)

        # print('sig', sig.size())
        print('SIG', sig)
        # TODO randomly the sig is NAN (?!)
        return sig

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

    def forward_sig(self, x, adj):
        # x in batch_size x n_samples x node_feats
        # g_out1 in batch_size x n_samples x g_hidden_size
        # g_out2 in batch_size x n_samples x nclass
        g_out1 = F.relu(self.gc1(x, adj))
        g_out1 = F.dropout(g_out1, self.dropout, training=self.training)
        g_out2 = F.relu(self.gc2(g_out1, adj))
        g_out2 = F.dropout(g_out2, self.dropout, training=self.training)

        print('x',x.size())
        print('g_out1', g_out1.size())
        print('g_out2', g_out2.size())
        sys.exit(0)

##########################################################################
##########################################################################
if __name__ == '__main__': main()
##########################################################################
##########################################################################

### Bayesian hard attention via stochastic graph sampling ###
# conda create -n attn python=3.6 scipy pytorch torchvision
# source activate attn
