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

import numpy as np
import tensorflow as tf
from prox_res_flow import *
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
import os


forward_model=lambda x: x[:,:1]
noise_level=0.1
def generate(n):
    x=tf.random.normal((n,2))
    x_norm=tf.math.sqrt(tf.reduce_sum(x**2,1))
    x=x/x_norm[:,tf.newaxis]+noise_level*tf.random.normal(x.shape)
    y=forward_model(x)
    y=y+0.02*tf.random.normal(y.shape)
    return x,forward_model(x)


x,y=generate(80000)

def orth_penalty():
    out=0.
    for netw in model.subnetworks:
        for s in netw.stiefel:
            if s.matrix.shape[0]>s.matrix.shape[1]:
                out+=tf.reduce_sum((tf.matmul(tf.transpose(s.matrix),s.matrix)-tf.eye(s.matrix.shape[1]))**2)
            else:
                out+=tf.reduce_sum((tf.matmul(s.matrix,tf.transpose(s.matrix))-tf.eye(s.matrix.shape[0]))**2)
    return out


batch_size=800
res_blocks=20
rep=64
def condition_network():
    net=tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(64,activation='relu'))
    net.add(tf.keras.layers.Dense(y.shape[1]))
    return net
model=ProxResFlow(res_blocks,64,3,actnorm=True,reproduce=rep,conditional=True,condition_network=condition_network,factor_init=2.,gamma=1.99)
model(tf.random.normal((5,2)),condition=y[:5])
for i in range(res_blocks):
    model.condition_networks[i].layers[-1].kernel.assign(rep*model.condition_networks[i].layers[-1].kernel)
model.actnorm_init(x,condition=y,batch_size=batch_size)
x_test=x[:2]
pred,ld=model(x_test,comp_logdet=True,condition=y[:2])
print(ld)
print(orth_penalty())

optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)

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

for netw in model.subnetworks:
    for s in netw.stiefel:
        s.project=False
        orth_s_matrix=proj_orth(s.matrix[None,:,:],num_iter=40)
        s.matrix.assign(tf.squeeze(orth_s_matrix))

def myplot(condition):
    n_plot=10000
    y_testies = tf.zeros((n_plot, y.shape[1]))
    y_testies = y_testies+condition
    outie = model.call_inverse(tf.random.normal((n_plot,2)),  condition= y_testies)
    outie = outie.numpy()
    pplot(outie,condition,0)
    pplot(outie,condition,1)

def pplot(outie,condition,dim):
    fig=plt.figure()
    n_bins = 100
    bins = np.linspace(-2, 2, n_bins + 1, endpoint=True)
    _, _, patches = plt.hist(outie[:,dim], bins)
    if not os.path.isdir('circle_imgs'+str(dim)):
        os.mkdir('circle_imgs'+str(dim))
    file_name='circ'+str(condition)
    file_name=file_name.replace('.','_')
    fig.savefig('circle_imgs'+str(dim)+'/'+file_name+'.png',dpi=1200)
    plt.close(fig)

for condition in [1.,0.7,0.,-0.7,-1.]:
    myplot(condition)

x,y=generate(80000)
fig=plt.figure()
plt.scatter(x[:10000,0],x[:10000,1], s=0.25)
plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.axis('off')
plt.tight_layout()
fig.savefig('circle_prior.png', bbox_inches='tight',pad_inches = 0.05)
plt.close(fig)

