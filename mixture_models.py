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
# This script reproduceds the mixture example from Section 4.2 of the paper.

import numpy as np
import tensorflow as tf
from prox_res_flow import *
import pickle
from mixture_utils import *
import os

DIMENSION=50
testing_num_y=10
b=0.05
n_mixtures=5

np.random.seed(20)
mixture_params=[]
# create mixture params (weights, means, covariances)
for i in range(n_mixtures):
    mixture_params.append((1./n_mixtures,np.random.uniform(size=DIMENSION)*2-1,0.01))

# draws testing_ys
forward_map = create_forward_model(scale = 0.1,dimension=DIMENSION)
forward_operator= lambda x: forward_pass(x,forward_map)
testing_xs = draw_mixture_dist(mixture_params, testing_num_y)
testing_ys = forward_pass(testing_xs, forward_map) + b * tf.random.normal((testing_num_y, DIMENSION))

def generate(n,test=False):
    dat_x = tf.constant(draw_mixture_dist(mixture_params, n),dtype=tf.float32)
    dat_y = forward_operator(dat_x)
    dat_y += tf.random.normal(dat_y.shape) * b
    return dat_x,dat_y

x,y=generate(10000)


def orth_penalty():
    out=0.
    for netw in model.subnetworks:
        for s in netw.stiefel:
            if s.matrix.shape[0]>s.matrix.shape[1]:
                out+=tf.reduce_sum((tf.matmul(tf.transpose(s.matrix),s.matrix)-tf.eye(s.matrix.shape[1]))**2)
            else:
                out+=tf.reduce_sum((tf.matmul(s.matrix,tf.transpose(s.matrix))-tf.eye(s.matrix.shape[0]))**2)
    return out

batch_size=200
res_blocks=20
rep=2

def condition_network():
    net=tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(256,activation='relu'))
    net.add(tf.keras.layers.Dense(y.shape[1]))
    return net

model=ProxResFlow(res_blocks,128,3,actnorm=True,reproduce=rep,conditional=True,condition_network=condition_network,factor_init=2.,gamma=1.99)
model(tf.random.normal((5,DIMENSION)),condition=tf.constant(y[:5],dtype=tf.float32))



model.actnorm_init(tf.constant(x,dtype=tf.float32),condition=tf.constant(y,dtype=tf.float32),batch_size=batch_size)
x_test=testing_xs[:2]
pred,ld=model(x_test,comp_logdet=True,condition=tf.constant(y[:2],dtype=tf.float32))
optimizer=tf.keras.optimizers.Adam(learning_rate=5e-3)

@tf.function
def train_step(xs,ys):
    with tf.GradientTape() as tape:
        preds,my_logdet=model(xs,condition=ys,comp_logdet=True)
        logpz=.5*tf.reduce_sum(preds**2)/xs.shape[0]
        logdet=-my_logdet/xs.shape[0]
        orth=orth_penalty()
        out=logpz+logdet+orth


    grad=tape.gradient(out,model.trainable_variables)
    optimizer.apply_gradients(zip(grad,model.trainable_variables))
    return out,logpz,logdet,orth


epch=20
steps_per_epch=2000
for ep in range(epch):
    obj=0.
    obj_pz,obj_logdet,obj_orth=[0.,0.,0.]
    for counter in range(1,steps_per_epch+1):
        xs,ys=generate(batch_size)
        out=train_step(xs,ys)
        obj+=out[0]
        obj_pz+=out[1]
        obj_logdet+=out[2]
        obj_orth+=out[3]
        if counter%100==0:
            print(counter)
            print('obj_pz,obj_logdet,obj_orth:',(obj_pz/counter).numpy(),(obj_logdet/counter).numpy(),(obj_orth/counter).numpy(),sep=', ')
    obj_without_orth=((obj-obj_orth)/steps_per_epch).numpy()
    print('Epoch, loss, orth, loss without orth:',ep+1,(obj/steps_per_epch).numpy(),orth_penalty().numpy(),obj_without_orth,sep=', ')
    print('obj_pz,obj_logdet,obj_orth:',(obj_pz/steps_per_epch).numpy(),(obj_logdet/steps_per_epch).numpy(),(obj_orth/steps_per_epch).numpy(),sep=', ')

model.save_weights('mixture_weights')

