import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import animation
import seaborn as sns
sns.set(color_codes=True)

def main():
    gan = GAN()
    gan.label = 2
    gan.x_train = read_data(gan.label)
    gan.input_size = gan.x_train.shape[1]
    gan.batch_size = 30
    gan.noise_range = 1.
    gan.data_learning_rate = 0.1
    gan.gen_learning_rate = 0.1
    gan.nn_std = .01
    gan.nn_bias = 0.1
    gan.num_steps = 200
    gan.create_model()
    gan.train()
    
class NeuralNet:
    def __init__(self):
        self.n_hidden = 50
        self.input_size = None
        self.output_size = 10
        self.scope = None
        self.std = 0.1 # weight initialization std
        self.bias = 0.1 # bias initialization
        
        
        #w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=norm)
        #b = tf.get_variable('b', [output_dim], initializer=const)
        
    def initialize_variables(self):
        norm = tf.random_normal_initializer(stddev=self.std)
        const = tf.constant_initializer(self.bias)
        with tf.variable_scope(self.scope or 'linear'):
            self.W1 = tf.get_variable('W1', [self.input_size, self.n_hidden], initializer=norm)
            #self.W2 = tf.get_variable('W2', [self.n_hidden, self.n_hidden], initializer=norm)
            self.b1 = tf.get_variable('b1', [self.n_hidden], initializer = const)
            #self.b2 = tf.get_variable('b2', [self.n_hidden], initializer = const)
            if self.scope == "D":
                self.W2 = tf.get_variable('W2', [self.n_hidden, self.n_hidden], initializer=norm)
                self.b2 = tf.get_variable('b2', [self.n_hidden], initializer = const)
                self.W3 = tf.get_variable('W3', [self.n_hidden, self.n_hidden], initializer=norm)
                self.W4 = tf.get_variable('W4', [self.n_hidden, self.output_size], initializer=norm)
                self.b3 = tf.get_variable('b3', [self.n_hidden], initializer = const)
                self.b4 = tf.get_variable('b4', [self.output_size], initializer = const)
            elif self.scope == "G":
                self.W2 = tf.get_variable('W2', [self.n_hidden, self.input_size], initializer=norm)
                self.b2 = tf.get_variable('b2', [self.input_size], initializer = const)
                #self.W3 = tf.get_variable('W3', [self.n_hidden, self.input_size], initializer=norm)
                #self.b3 = tf.get_variable('b3', [self.input_size], initializer = const)
            
    def mlp(self, input, mode):
        if mode == "gen":
            z1 = tf.matmul(input, self.W1) + self.b1
            a1 = tf.nn.tanh(z1)
            z2 = tf.matmul(a1, self.W2) + self.b2
            #a2 = tf.nn.tanh(z2)
            #z3 = tf.matmul(a2, self.W3) + self.b3
            return tf.nn.tanh(z2)
        elif mode == "disc":
            z1 = tf.matmul(input, self.W1) + self.b1
            a1 = tf.nn.relu(z1)
            z2 = tf.matmul(a1, self.W2) + self.b2
            a2 = tf.nn.relu(z2)
            z3 = tf.matmul(a2, self.W3) + self.b3
            a3 = tf.nn.relu(z3)
            z4 = tf.matmul(a3, self.W4) + self.b4
            return tf.nn.sigmoid(z4), z4
        else:
            print "Mode not valid"
class GAN:
    def __init__(self):
        self.G = None
        self.D1 = None
        self.D2 = None
        self.D1_logits = None
        self.D2_logits = None
        self.data_mu = -2.
        self.data_std = .5
        self.num_steps = 5000
        self.input_size = None
        self.batch_size = None
        self.z = None
        self.x = None
        self.x_train = None
        self.loss_d = None
        self.loss_g = None
      # gan.d_pre_params = [v for v in vars if v.name.startswith('D_pre/')]
        self.d_params = None
        self.g_params = None
    
        self.opt_d = None
        self.opt_g = None
        self.nn_bias = None
        self.nn_std = None
        self.anim_frames = []
    
    
    def generator(self, input):
        nn_gen = NeuralNet()
        nn_gen.input_size = self.input_size
        nn_gen.scope = "G"
        nn_gen.bias = self.nn_bias
        nn_gen.std = self.nn_std
        nn_gen.initialize_variables()
        self.nn = nn_gen
        G = self.nn.mlp(input, "gen")
        return G
    
    def discriminator(self, input):
        nn_disc = NeuralNet()
        nn_disc.input_size = self.input_size
        nn_disc.scope = "D"
        nn_disc.initialize_variables()
        self.nn = nn_disc
        D, D_logits = self.nn.mlp(input, "disc")
        return D, D_logits
        
    def noise_samples(self):
        asd = np.random.uniform(-self.noise_range, self.noise_range, size = (self.batch_size, self.input_size))
        return asd
        
    def create_model(self):
        tf.reset_default_graph()
        with tf.variable_scope('G'):
            z = tf.placeholder(tf.float32, shape=(None, self.input_size))
            G  = self.generator(z)
            self.G = G
            self.z = z
            
        with tf.variable_scope('D') as scope:
            x = tf.placeholder(tf.float32, shape=(None, self.input_size))
            self.D1, self.D1_logits = self.discriminator(x)
            scope.reuse_variables()
            self.D2, self.D2_logits = self.discriminator(G)
            self.x = x
        
        self.loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D1_logits, tf.ones_like(self.D1))\
            + tf.nn.sigmoid_cross_entropy_with_logits(self.D2_logits, tf.zeros_like(self.D2)))
        self.loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D2_logits, tf.ones_like(self.D2)))
    
        vars = tf.trainable_variables()
        # gan.d_pre_params = [v for v in vars if v.name.startswith('D_pre/')]
        self.d_params = [v for v in vars if v.name.startswith('D/')]
        print [v.name for v in vars]
        self.g_params = [v for v in vars if v.name.startswith('G/')]
    
        self.opt_d = optimizer(self.data_learning_rate, self.loss_d, self.d_params)
        self.opt_g = optimizer(self.gen_learning_rate, self.loss_g, self.g_params)    
                

    def train(self):
        session = tf.InteractiveSession()
        session.run(tf.initialize_all_variables())

        loss_d_array = []
        loss_g_array = []
        np.random.shuffle(self.x_train)
        for ite in range(self.num_steps):
            for step in xrange(self.x_train.shape[0]//self.batch_size):
                # update discriminator
                x1 = self.x_train[step*self.batch_size:((step+1)*self.batch_size),:]
                z1 = self.noise_samples()
                
                loss_d, _ = session.run([self.loss_d, self.opt_d], {
                    self.x: x1,
                    self.z: z1 
                    })
    
                # update generator
                z1 = self.noise_samples()
                loss_g, _ = session.run([self.loss_g, self.opt_g], {
                    self.z: z1
                })
    
                if step % 20 == 0:
                    print('Epoch {}, Batch {}: {}\t{}'.format(ite, step, loss_d, loss_g))
                    loss_d_array.append(loss_d)
                    loss_g_array.append(loss_g)
                    #print([v.eval() for v in tf.all_variables() if v.name == "G/G/W1:0"])
                    #print([v.eval() for v in tf.all_variables() if v.name == "D/D/W1:0"])
        
            #session.close()
        
        zz = self.G.eval(feed_dict = {self.z: z1})[0:16,:]
        self._plot_loss(loss_d_array, loss_g_array)
        tiled_im = img_tile(np.reshape(zz, [zz.shape[0], 28, 28]), aspect_ratio=.5, tile_shape=[4,4], border=0, border_color=0, stretch=False)
        plt.figure()
        plt.imshow(tiled_im)
        plt.grid(False)
        plt.show()
        plt.savefig("image_L%d_DLR%f_GLR%f_BS%d_N%d.png"%(self.label, self.data_learning_rate, self.gen_learning_rate, self.batch_size, self.num_steps))
        session.close()

    def _plot_loss(self, loss_d_array, loss_g_array):
        xx = np.linspace(1,len(loss_d_array))*20
        plt.figure()
        plt.plot(loss_d_array)
        plt.plot(loss_g_array)
        plt.legend(['Discriminator', 'Generator'])
        plt.title("Training loss with Data LR = %f, Gen LR = %f"%(self.data_learning_rate, self.gen_learning_rate))
        plt.show()
        plt.savefig("loss_L%d_DLR%f_GLR%f_BS%d_N%d.png"%(self.label, self.data_learning_rate, self.gen_learning_rate, self.batch_size, self.num_steps))
        

def optimizer(learning_rate, loss, var_list):
    initial_learning_rate = learning_rate
    decay = 0.95
    num_decay_steps = 100
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_steps,
        decay,
        staircase=True
    )
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss,
        global_step=batch,
        var_list=var_list
    )
    return optimizer
    

def read_data(label):   
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        
    x_train = mnist.train.images
    y_train = mnist.train.labels
    x_test = mnist.test.images
    y_test = mnist.test.labels
    
    print x_train.shape
    print y_train.shape
    print x_test.shape
    print y_test.shape
    return x_train[np.argmax(y_train, axis = 1)==label,:]
   

def img_tile(imgs, aspect_ratio=1.0, tile_shape=None, border=1,
             border_color=0, stretch=False):
    ''' Tile images in a grid.
    If tile_shape is provided only as many images as specified in tile_shape
    will be included in the output.
    '''

    # Prepare images
    #if stretch:
    #    imgs = img_stretch(imgs)
    imgs = np.array(imgs)
    if imgs.ndim != 3 and imgs.ndim != 4:
        raise ValueError('imgs has wrong number of dimensions.')
    n_imgs = imgs.shape[0]

    # Grid shape
    img_shape = np.array(imgs.shape[1:3])
    if tile_shape is None:
        img_aspect_ratio = img_shape[1] / float(img_shape[0])
        aspect_ratio *= img_aspect_ratio
        tile_height = int(np.ceil(np.sqrt(n_imgs * aspect_ratio)))
        tile_width = int(np.ceil(np.sqrt(n_imgs / aspect_ratio)))
        grid_shape = np.array((tile_height, tile_width))
    else:
        assert len(tile_shape) == 2
        grid_shape = np.array(tile_shape)

    # Tile image shape
    tile_img_shape = np.array(imgs.shape[1:])
    tile_img_shape[:2] = (img_shape[:2] + border) * grid_shape[:2] - border

    # Assemble tile image
    tile_img = np.empty(tile_img_shape)
    tile_img[:] = border_color
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            img_idx = j + i*grid_shape[1]
            if img_idx >= n_imgs:
                # No more images - stop filling out the grid.
                break
            img = imgs[img_idx]
            yoff = (img_shape[0] + border) * i
            xoff = (img_shape[1] + border) * j
            tile_img[yoff:yoff+img_shape[0], xoff:xoff+img_shape[1], ...] = img

    return tile_img

main() 