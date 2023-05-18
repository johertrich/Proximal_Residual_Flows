# This code belongs to the paper
#
# J. Hertrich
# Proximal Residual Flows for Bayesian Inverse Problems.
# L. Calatroni, M. Donatelli, S. Morigi, M. Prato and M. Santacesaria (eds.)
# Scale Space and Variational Methods in Computer Vision.
# Lecture Notes in Computer Science, 14009, 210-222, 2023.
#
# Paper available at https://doi.org/10.1007/978-3-031-31975-4_16
# Preprint available at https://arxiv.org/abs/2211.17158
#
# Please cite the paper, if you use this code.
# This script provides the general implementation of proximal residual flows.
#

import tensorflow as tf
import numpy as np

def proj_orth(X,num_iter=4):
    # Projects the matrices X[...,:,:] onto the orthogonal Stiefel manifold.
    trans=False
    if X.shape[0]<X.shape[1]:
        X=tf.transpose(X,perm=[0,2,1])
        trans=True
    Y=X
    for i in range(num_iter):
        Y_inv=tf.eye(X.shape[2],dtype=tf.float32,batch_shape=[X.shape[0]])+tf.matmul(Y,Y,transpose_a=True)
        Y=2*tf.matmul(Y,tf.linalg.inv(Y_inv))
    if trans:
        Y=tf.transpose(Y,perm=[0,2,1])
    return Y

class StiefelDenseLayer(tf.keras.layers.Layer):
    # Implements a Layer of the form A^T sigma(A x+ b) where sigma is an activation function for two-dimensional
    # data points.
    # Inputs:   num_outputs     = number of hidden neurons = dim(Ax)
    #           activation      = activation function for the layer. None for soft shrinkage
    def __init__(self,num_outputs,activation):
        super(StiefelDenseLayer, self).__init__()
        self.activation=activation
        self.num_outputs=num_outputs
        self.project=True


    def build(self, input_shape):
        self.matrix=self.add_weight("matrix",initializer='orthogonal',shape=[self.num_outputs,int(np.prod(input_shape[1:]))],trainable=True)
        self.bias=self.add_weight("bias",initializer='zeros', shape=[self.num_outputs], trainable=True)

    def call(self, inputs):
        inp_shape=inputs.shape
        inputs=tf.reshape(inputs,[-1,tf.reduce_prod(inputs.shape[1:])])
        matrix=self.matrix
        if self.project:
            matrix=tf.squeeze(proj_orth(matrix[None,:,:]))
        x=tf.linalg.matmul(matrix,tf.transpose(inputs))
        x = tf.nn.bias_add(tf.transpose(x), self.bias)
        x=self.activation(x)
        outputs = tf.transpose(tf.linalg.matmul(tf.transpose(matrix),tf.transpose(x)))
        outputs=tf.reshape(outputs,inp_shape)
        return outputs

class ActNorm(tf.keras.layers.Layer):
    def __init__(self):
        super(ActNorm,self).__init__()
    def build(self,input_shape):
        self.scale=self.add_weight("scale",initializer='random_normal',shape=input_shape[1:],trainable=True)
        self.bias=self.add_weight("bias",initializer='zeros',shape=input_shape[1:],trainable=True)
    def call(self,inputs):
        x=tf.multiply(inputs,tf.exp(self.scale))
        x=tf.add(x,self.bias)
        return x

class StiefelNetwork(tf.keras.Model):
    # Implements a PNN.
    def __init__(self,hidden_dim,n_hidden_layers,activation):
        super(StiefelNetwork,self).__init__()
        self.stiefel=[]
        for i in range(n_hidden_layers):
            self.stiefel.append(StiefelDenseLayer(hidden_dim,activation))

    def call(self,x):
        for layer in self.stiefel:
            x=layer(x)
        return x

class ProxResFlow(tf.keras.Model):
    # Implements a sequence of residual blocks with PNNs.
    def __init__(self,res_blocks,hidden_dim,n_hidden_layers,activation=tf.keras.activations.elu,actnorm=False,reproduce=1,conditional=False,condition_network=None,factor_init=1.,gamma=1.):
        super(ProxResFlow,self).__init__()
        self.res_blocks=res_blocks
        self.subnetworks=[]
        if type(activation)!=list:
            activation=[activation]*res_blocks
        if type(hidden_dim)!=list:
            hidden_dim=[hidden_dim]*res_blocks
        self.actnorm=actnorm
        self.actnorm_layers=[]
        if type(reproduce)!=list:
            reproduce=[reproduce]*res_blocks
        self.reproduce=reproduce
        self.conditional=conditional
        self.gamma=gamma
        self.factor_init=factor_init
        self.factors=None
        self.condition_networks=[]
        self.condition_network_constructor=condition_network
        self.condition_network=not (condition_network is None)
        for i in range(res_blocks):
            self.subnetworks.append(StiefelNetwork(hidden_dim[i],n_hidden_layers,activation[i]))
            if self.actnorm:
                self.actnorm_layers.append(ActNorm())
            if condition_network is None:
                self.condition_networks.append(lambda x:x)
            else:
                self.condition_networks.append(condition_network())

    def call(self,x,comp_logdet=False,condition=None,outer_tape=None,interm_steps=None,training=True):
        if self.factors is None:
            self.factors=tf.Variable(self.factor_init*tf.ones(self.res_blocks))
        if self.conditional and condition is None:
            raise ValueError('Condition required!')
        logdet_sum=0.
        for i,network in enumerate(self.subnetworks):
            network_fun=self.get_network_fun(i,condition)
            if comp_logdet:
                x_shape=x.shape
                x_flat=tf.reshape(x,[x.shape[0],-1])
                with tf.GradientTape() as tape:
                    tape.watch(x_flat)
                    x=tf.reshape(x_flat,x_shape)
                    net_out=network_fun(x)
                    net_out_flat=tf.reshape(net_out,[x.shape[0],-1])
                jacobians=self.gamma*tape.batch_jacobian(net_out_flat,x_flat)+tf.eye(x_flat.shape[1],batch_shape=[x.shape[0]])
                _,ld=tf.linalg.slogdet(jacobians)
                logdet_sum+=tf.reduce_sum(ld)
            else:
                net_out=network_fun(x)
            x=x+self.gamma*net_out
            if self.actnorm:
                x=self.actnorm_layers[i](x)
                if comp_logdet:
                    logdet_sum+=tf.reduce_sum(self.actnorm_layers[i].scale)*x.shape[0]
        if comp_logdet:
            return x,logdet_sum
        return x
    
    def call_inverse(self,y,condition=None):
        eps=1e-10
        for i in range(1,len(self.subnetworks)+1):
            if self.actnorm:
                y=(y-self.actnorm_layers[-i].bias)/tf.exp(self.actnorm_layers[-i].scale)
            network=self.subnetworks[-i]
            network_fun=self.get_network_fun(-i,condition=condition)
            n_layer=1.*len(network.stiefel)
            t=n_layer/(n_layer+1)
            y_=y/(1+self.gamma-self.gamma*t)
            x=tf.identity(y_)
            tR=lambda z: (network_fun(z)-(1-t)*z)
            for itera in range(500):
                x_new=y_-self.gamma*tR(x)/(1+self.gamma-self.gamma*t)
                change=tf.reduce_sum((x_new-x)**2)/x.shape[0]
                x=x_new
                if itera%100==0:
                    print(change.numpy())
                if change<eps:
                    break
            if change>eps:
                print('Banach iteration did not converge in layer '+str(len(self.subnetworks)-i)+' with criteria '+str(change.numpy()))
            y=x
        return x

    def get_network_fun(self,i,condition=None):
        network=self.subnetworks[i]
        if self.reproduce[i]==1:
            if self.conditional:
                network_fun=lambda y: network(tf.concat([y,self.condition_networks[i](condition)],-1))[:,:y.shape[-1]]
            else:
                network_fun=network
        else:
            if self.conditional:
                def network_fun(y):
                    net_out=network(tf.concat([tf.tile(y,(1,self.reproduce[i])),self.condition_networks[i](condition)],-1)[:,:,tf.newaxis])[:,:-condition.shape[1]]
                    net_out=tf.reduce_sum(tf.reshape(net_out,[net_out.shape[0],self.reproduce[i],-1]),axis=1)
                    return net_out/self.reproduce[i]
            else:
                network_fun=lambda y:tf.reduce_sum(network(tf.repeat(tf.expand_dims(y,-1),self.reproduce[i],-1)),axis=-1)/self.reproduce[i]
        network_fun2=lambda y: network_fun(tf.exp(self.factors[i])*y)/(tf.exp(self.factors[i]))
        return network_fun2

    def actnorm_init(self,data,condition=None,batch_size=100,mean=0,std=1.):
        if not self.actnorm:
            return
        if self.conditional:
            self(data[:5],condition=condition[:5])
        else:
            self(data[:5])
        x=data
        for i,network in enumerate(self.subnetworks):
            print(i)
            network_fun=self.get_network_fun(i,condition=condition)
            if self.conditional:
                net_ds=tf.data.Dataset.from_tensor_slices((x,condition)).batch(batch_size)
                net_out=[]
                for data_batch,condition_batch in net_ds:
                    network_fun=self.get_network_fun(i,condition=condition_batch)
                    net_out.append(network_fun(data_batch))
            else:
                net_ds=tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
                net_out=[]
                for j,data_batch in enumerate(net_ds):
                    net_out.append(network_fun(data_batch))
            net_out=tf.concat(net_out,0)
            x=x+self.gamma*net_out
            pointwise_mean=tf.reduce_mean(x,axis=0)
            pointwise_std=tf.math.reduce_std(tf.subtract(x,pointwise_mean),axis=0)/std
            self.actnorm_layers[i].scale.assign(tf.math.log(1./pointwise_std))
            self.actnorm_layers[i].bias.assign(-pointwise_mean/pointwise_std+mean)
            x=self.actnorm_layers[i](x)

