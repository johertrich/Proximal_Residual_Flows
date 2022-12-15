# This code belongs to the paper
#
# J. Hertrich
# Proximal Residual Flows or Bayesian Inverse Problems
# Arxiv Preprint 2211.17158
#
# Please cite the paper, if you use this code.
# This script reproduceds the toy examples from Section 4.1 of the paper.

import numpy as np
import tensorflow as tf
import sklearn.datasets
import matplotlib.pyplot as plt
import os

dataset='circles'

if dataset=='modes':
    verts = [
             (-2.4142, 1.),
             (-1., 2.4142),
             (1.,  2.4142),
             (2.4142,  1.),
             (2.4142, -1.),
             (1., -2.4142),
             (-1., -2.4142),
             (-2.4142, -1.)
            ]


    def generate(n):
        cum_alpha=np.cumsum(np.ones(8)/8.)
        zv=np.random.uniform(size=(n))
        classes=np.zeros(n)
        for i in range(0,7):
            classes=classes+(zv>cum_alpha[i])
        pos = np.random.normal(size=(n, 2), scale=0.2)
        for i, v in enumerate(verts):
            pos[classes==i,:]+=v
        return tf.constant(pos,dtype=tf.float32)
    ax_range=[-3.5,3.5,-3.5,3.5]
    n_layers=20
    rep=64
elif dataset=='checkboard':
    def generate(n):
        cum_alpha=np.cumsum(np.ones(2)/2.)
        zv=np.random.uniform(size=(n))
        classes=np.zeros(n)
        for i in range(0,1):
            classes=classes+(zv>cum_alpha[i])
        pos = np.random.uniform(size=(n, 2))/2.
        pos[classes==1,:]+=(0.5,.5)
        return tf.constant(pos,dtype=tf.float32)
    ax_range=[0.,1.,0.,1.]
    n_layers=20
    rep=64
elif dataset=='moons':
    def generate(n):
        pos,_=sklearn.datasets.make_moons(n_samples=n,noise=0.05)
        return tf.constant(pos,dtype=tf.float32)
    ax_range=[-1.5,2.5,-2.,2.]
    n_layers=20
    rep=64
elif dataset=='circles':
    def generate(n):
        pos,_=sklearn.datasets.make_circles(n_samples=n, factor=0.5, noise=0.05)
        return tf.constant(pos,dtype=tf.float32)
    ax_range=[-1.5,1.5,-1.5,1.5]
    n_layers=20
    rep=64
else:
    raise ValueError('Dataset not known!')


from prox_res_flow import *

def orth_penalty():
    out=0.
    for netw in model.subnetworks:
        for s in netw.stiefel:
            if s.matrix.shape[0]>s.matrix.shape[1]:
                out+=tf.reduce_sum((tf.matmul(tf.transpose(s.matrix),s.matrix)-tf.eye(s.matrix.shape[1]))**2)
            else:
                out+=tf.reduce_sum((tf.matmul(s.matrix,tf.transpose(s.matrix))-tf.eye(s.matrix.shape[0]))**2)
    return out

model=ProxResFlow(n_layers,64,3,actnorm=True,reproduce=rep,factor_init=2.,gamma=1.99)
model.actnorm_init(generate(10000))
x_test=tf.random.normal((2,2))
pred,ld=model(x_test,comp_logdet=True)
print(ld)
x_recon=model.call_inverse(pred)
print(orth_penalty())
print(tf.reduce_sum((x_test-x_recon)**2))
data=generate(100000)
learning_rate=1e-3
optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)

@tf.function
def train_step(xs):
    with tf.GradientTape() as tape:
        preds,my_logdet=model(xs,comp_logdet=True)
        logpz=.5*tf.reduce_sum(preds**2)/xs.shape[0]
        logdet=-my_logdet/xs.shape[0]
        orth=orth_penalty()
        out=logpz+logdet+orth
    grads=tape.gradient(out,model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))
    return out,logpz,logdet,orth

batch_size=200
epch=20
steps_per_epch=2000
for ep in range(epch):
    obj=0.
    obj_pz,obj_logdet,obj_orth=[0.,0.,0.]
    for counter in range(1,steps_per_epch+1):
        xs=generate(batch_size)
        out=train_step(xs)
        obj+=out[0]
        obj_pz+=out[1]
        obj_logdet+=out[2]
        obj_orth+=out[3]
        if counter%100==0:
            print(counter)
            print('obj_pz,obj_logdet,obj_orth:',(obj_pz/counter).numpy(),(obj_logdet/counter).numpy(),(obj_orth/counter).numpy(),sep=', ')
    print('Epoch, loss, orth, loss without orth:',ep+1,(obj/steps_per_epch).numpy(),orth_penalty().numpy(),((obj-obj_orth)/steps_per_epch).numpy(),sep=', ')
    print('obj_pz,obj_logdet,obj_orth:',(obj_pz/steps_per_epch).numpy(),(obj_logdet/steps_per_epch).numpy(),(obj_orth/steps_per_epch).numpy(),sep=', ')

for netw in model.subnetworks:
    for s in netw.stiefel:
        s.project=False
        orth_s_matrix=proj_orth(s.matrix[None,:,:],num_iter=40)
        s.matrix.assign(tf.squeeze(orth_s_matrix))
n_samples = 20000
y_test=tf.random.normal((n_samples,2))
preds=model.call_inverse(y_test)


if not os.path.isdir('toy_results'):
    os.mkdir('toy_results')

y_test=tf.random.normal((10000,2))
preds=model.call_inverse(y_test)
preds2=model(data[:10000])

fig=plt.figure()
plt.scatter(data[:10000,0],data[:10000,1], s=0.25)
plt.xlim(ax_range[0], ax_range[1])
plt.ylim(ax_range[2], ax_range[3])
plt.gca().set_aspect('equal', adjustable='box')
plt.axis('off')
plt.tight_layout()
fig.savefig('toy_results/'+dataset+'_gt.png', bbox_inches='tight',pad_inches = 0.05)
plt.close(fig)

fig=plt.figure()
plt.scatter(preds[:10000,0],preds[:10000,1], s=0.25)
plt.xlim(ax_range[0], ax_range[1])
plt.ylim(ax_range[2], ax_range[3])
plt.gca().set_aspect('equal', adjustable='box')
plt.axis('off')
plt.tight_layout()
fig.savefig('toy_results/'+dataset+'_recon.png', bbox_inches='tight',pad_inches = 0.05)
plt.close(fig)

fig, axes = plt.subplots(1, 4, figsize=(16,4))


axes[0].clear()
axes[0].scatter(data[:10000,0],data[:10000,1], s=0.25)
axes[0].axis(ax_range)

axes[1].clear()
axes[1].scatter(preds2[:,0],preds2[:,1], s=0.25)
axes[1].axis([-3.5,3.5,-3.5,3.5])

axes[2].clear()
axes[2].scatter(y_test[:,0],y_test[:,1], s=0.25)
axes[2].axis([-3.5,3.5,-3.5,3.5])

axes[3].clear()
axes[3].scatter(preds[:,0],preds[:,1], s=0.25)
axes[3].axis(ax_range)
fig.savefig(dataset+'.png',dpi=1200)
plt.show()
