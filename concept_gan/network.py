import tensorflow as tf
import numpy as np
from trainingdata import *
import seaborn as sb
import matplotlib.pyplot as plt
sb.set()

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

# input takes in the placeholder for random samples (Z), an array hsize for the number of units in the 2 hidden layers
# and reuse variable which is used for reusing the same layers. Using these inputs it creates a fully connected neural network of 2 hidden layers with given number of nodes. 
# The output of this function is a 2-dimensional vector which corresponds to the dimensions of the real dataset that we are trying to learn. 
# The above function can be easily modified to include more hidden layers, different types of layers, different activation and different output mappings.

def generator(Z,hsize=[16, 16],reuse=False):
    with tf.variable_scope("GAN/Generator",reuse=reuse):
        h1 = tf.layers.dense(Z,hsize[0],activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1,hsize[1],activation=tf.nn.leaky_relu)
        out = tf.layers.dense(h2,2)

    return out

# input placeholder for the samples from the vector space of real dataset. 
# The samples can be both real samples and samples generated from the Generator network. 
# Similar to the Generator network above it also takes input hsize and reuse. We use 3 hidden layers for the Discriminator out of which first 2 layers size we take input.
# We fix the size of the third hidden layer to 2 so that we can visualize the transformed feature space in a 2D plane as explained in the later section.
# The output of this function is a logit prediction for the given X and the output of the last layer which is the feature transformation learned by Discriminator for X. 
# The logit function is the inverse of the sigmoid function which is used to represent the logarithm of the odds (ratio of the probability of variable being 1 to that of it being 0)

def discriminator(X,hsize=[16, 16],reuse=False):
    with tf.variable_scope("GAN/Discriminator",reuse=reuse):
        h1 = tf.layers.dense(X,hsize[0],activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1,hsize[1],activation=tf.nn.leaky_relu)
        h3 = tf.layers.dense(h2,2)
        out = tf.layers.dense(h3,1)

    return out, h3

# real and random noise
X = tf.placeholder(tf.float32,[None,2])
Z = tf.placeholder(tf.float32,[None,2])

# create graph for genrating samples from generator network and feed real and generated samples to discriminator networks
# using functions and placeholders defined above

G_sample = generator(Z)
r_logits, r_rep = discriminator(X)
f_logits, g_rep = discriminator(G_sample,reuse=True)

# Using the logits for generated data and real data we define the loss functions for the Generator and Discriminator networks
disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits,labels=tf.ones_like(r_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.zeros_like(f_logits)))
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.ones_like(f_logits)))

# These losses are sigmoid cross entropy based losses using the equations we defined above.
# commonly used loss function for so-called discrete classification
# inputs logit (given by discriminator network) and true labels for each sample <- then calcualtes the error for each sample
# on Tensorflow using optimized version, more stable than directly calc cross entropy

# optimisers def using loss functions, RMSOptimizer learning rate =0.001

gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")

gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss,var_list = gen_vars) # G Train step
disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss,var_list = disc_vars) # D Train step



# ====================tensorflow=========================

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

batch_size = 256
nd_steps = 10
ng_steps = 10

x_plot = sample_data(n=batch_size)

f = open('loss_logs.csv','w')
f.write('Iteration,Discriminator Loss,Generator Loss\n')
# =====================================================


# train the network
for i in range(100001):
    X_batch = sample_data(n=batch_size)
    Z_batch = sample_Z(batch_size, 2)

    # one run through the discriminator network
    for _ in range(nd_steps):
        _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: X_batch, Z: Z_batch})
    rrep_dstep, grep_dstep = sess.run([r_rep, g_rep], feed_dict={X: X_batch, Z: Z_batch})

    # one run through the generator network
    for _ in range(nd_steps):
        _, gloss = sess.run([gen_step, gen_loss], feed_dict={Z: Z_batch})
    rrep_gstep, grep_gstep = sess.run([r_rep, g_rep], feed_dict={X: X_batch, Z: Z_batch})

    # print instr per 1000 iterations
    print("Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f"%(i,dloss,gloss))
    if i%10 == 0:
        f.write("%d,%f,%f\n"%(i,dloss,gloss))

    if i%1000 == 0:
        plt.figure()
        g_plot = sess.run(G_sample, feed_dict={Z: Z_batch})
        xax = plt.scatter(x_plot[:,0], x_plot[:,1])
        gax = plt.scatter(g_plot[:,0],g_plot[:,1])

        plt.legend((xax,gax), ("Real Data","Generated Data"))
        plt.title('Samples at Iteration %d'%i)
        plt.tight_layout()
        plt.savefig('plots/iterations/iteration_%d.png'%i)
        plt.close()

        plt.figure()
        rrd = plt.scatter(rrep_dstep[:,0], rrep_dstep[:,1], alpha=0.5)
        rrg = plt.scatter(rrep_gstep[:,0], rrep_gstep[:,1], alpha=0.5)
        grd = plt.scatter(grep_dstep[:,0], grep_dstep[:,1], alpha=0.5)
        grg = plt.scatter(grep_gstep[:,0], grep_gstep[:,1], alpha=0.5)


        plt.legend((rrd, rrg, grd, grg), ("Real Data Before G step","Real Data After G step",
                               "Generated Data Before G step","Generated Data After G step"))
        plt.title('Transformed Features at Iteration %d'%i)
        plt.tight_layout()
        plt.savefig('plots/features/feature_transform_%d.png'%i)
        plt.close()

        plt.figure()

        rrdc = plt.scatter(np.mean(rrep_dstep[:,0]), np.mean(rrep_dstep[:,1]),s=100, alpha=0.5)
        rrgc = plt.scatter(np.mean(rrep_gstep[:,0]), np.mean(rrep_gstep[:,1]),s=100, alpha=0.5)
        grdc = plt.scatter(np.mean(grep_dstep[:,0]), np.mean(grep_dstep[:,1]),s=100, alpha=0.5)
        grgc = plt.scatter(np.mean(grep_gstep[:,0]), np.mean(grep_gstep[:,1]),s=100, alpha=0.5)

        plt.legend((rrdc, rrgc, grdc, grgc), ("Real Data Before G step","Real Data After G step",
                               "Generated Data Before G step","Generated Data After G step"))

        plt.title('Centroid of Transformed Features at Iteration %d'%i)
        plt.tight_layout()
        plt.savefig('plots/features/feature_transform_centroid_%d.png'%i)
        plt.close()

f.close()

# Generator Update
# Visualize the effect of updating the Generator network weights within the adversarial training process,
# -> by plotting the activations of the last hidden layer of Discriminator network.

# Visualizing the feature transformation function learned by the Discriminator network. 
# -> what our network learns so that the real and fake data are separable.
# Calculate centroids of the points before an after the generator update

# During final iterations transformed features of real and generated samples get mixed,
# Discrimnator should not ne able to distinguish
