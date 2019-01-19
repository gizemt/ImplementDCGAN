import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib import animation


def main():
    gan = GAN()
    gan.batch_size = 120
    gan.noise_range = 10.
    gan.data_learning_rate = 0.005
    gan.gen_learning_rate = 0.005
    gan.nn_std = .1
    gan.nn_bias = 0.
    gan.num_steps = 200*20
    gan.create_model()
    gan.train()

class NeuralNet:
    def __init__(self):
        # self.n_layer = 3
        self.n_hidden = 11
        self.input_size = 10
        self.output_size = 1
        self.scope = None
        self.std = .5# weight initialization std
        self.bias = 0.1 # bias initialization
        
    def initialize_variables(self):
        norm = tf.random_normal_initializer(stddev=self.std)
        const = tf.constant_initializer(self.bias)
        with tf.variable_scope(self.scope or 'linear'):
            self.W1 = tf.get_variable('W1', [self.input_size, self.n_hidden], initializer=norm)
            #self.W2 = tf.get_variable('W2', [self.n_hidden, self.output_size], initializer=norm)
            self.b1 = tf.get_variable('b1', [self.output_size], initializer = const)
            #self.b2 = tf.get_variable('b2', [self.n_hidden], initializer = const)
            if self.scope == "D":
                self.W2 = tf.get_variable('W2', [self.n_hidden, self.n_hidden], initializer=norm)
                self.W3 = tf.get_variable('W3', [self.n_hidden, self.output_size], initializer=norm)
                #self.W4 = tf.get_variable('W4', [self.n_hidden, self.output_size], initializer=norm)
                self.b2 = tf.get_variable('b2', [self.n_hidden], initializer = const)
                self.b3 = tf.get_variable('b3', [self.output_size], initializer = const)
                #self.b4 = tf.get_variable('b4', [self.output_size], initializer = const)
            elif self.scope == "G":
                self.W2 = tf.get_variable('W2', [self.n_hidden, self.output_size], initializer=norm)
                self.b2 = tf.get_variable('b2', [self.output_size], initializer = const)
            
    def mlp(self, input, mode):
        if mode == "gen":
            z1 = tf.matmul(input, self.W1) + self.b1
            a1 = tf.nn.tanh(z1)
            z2 = tf.matmul(a1, self.W2) + self.b2
            #a2 = tf.nn.tanh(z2)
            #z3 = tf.matmul(a2, self.W3) + self.b3
            return z2#tf.nn.tanh(z2)
        elif mode == "disc":
            z1 = tf.matmul(input, self.W1) + self.b1
            a1 = tf.nn.relu(z1)
            z2 = tf.matmul(a1, self.W2) + self.b2
            a2 = tf.nn.relu(z2)
            z3 = tf.matmul(a2, self.W3) + self.b3
            #a3 = tf.nn.relu(z3)
            #z4 = tf.matmul(a3, self.W4) + self.b4
            return tf.nn.softplus(z3)
        else:
            print "Mode not valid"
class GAN:
    def __init__(self):
        self.G = None
        self.D1 = None
        self.D2 = None
        self.data_mu = 3.
        self.data_std = .2
        self.num_steps = 5000
        self.z = None
        self.x = None
        self.loss_d = None
        self.loss_g = None
        self.d_params = None
        self.g_params = None
    
        self.opt_d = None
        self.opt_g = None
        self.nn_bias = None
        self.nn_std = None
        self.anim_frames = []
    
    
    def generator(self, input):
        nn_gen = NeuralNet()
        nn_gen.input_size = input.get_shape()[1]
        nn_gen.scope = "G"
        nn_gen.bias = self.nn_bias
        nn_gen.std = self.nn_std
        nn_gen.initialize_variables()
        self.nn = nn_gen
        G = self.nn.mlp(input, "gen")
        return G
    
    def discriminator(self, input):
        nn_disc = NeuralNet()
        nn_disc.input_size = input.get_shape()[1]
        nn_disc.scope = "D"
        nn_disc.initialize_variables()
        self.nn = nn_disc
        D = self.nn.mlp(input, "disc")
        return D
        
    def noise_samples(self, N):
        asd = np.random.uniform(-self.noise_range, self.noise_range, N)#
        return asd
        
    def data_samples(self, N):
        samples = np.random.normal(self.data_mu, self.data_std, N)
        samples.sort()
        return samples
        
    def create_model(self):
        tf.reset_default_graph()
        with tf.variable_scope('G'):
            z = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            G  = self.generator(z)
            self.G = G
            self.z = z
            
        with tf.variable_scope('D') as scope:
            x = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            self.D1 = self.discriminator(x)
            scope.reuse_variables()
            self.D2 = self.discriminator(G)
            self.x = x
        
        self.loss_d = tf.reduce_mean(-tf.log(self.D1) - tf.log(1 - self.D2))
        self.loss_g = tf.reduce_mean(-tf.log(self.D2))
    
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
        for step in xrange(self.num_steps):
            # discriminator
            bs = self.batch_size
            x1 = self.data_samples(bs)
            z1 = self.noise_samples(bs)
            
            loss_d, _ = session.run([self.loss_d, self.opt_d], {self.x: np.reshape(x1, (self.batch_size, 1)),self.z: np.reshape(z1, (self.batch_size, 1))})

            # generator
            z1 = self.noise_samples(bs)
            loss_g, _ = session.run([self.loss_g, self.opt_g], {self.z: np.reshape(z1, (self.batch_size, 1))})

            if step % 20 == 0:
                print('{}: {}\t{}'.format(step, loss_d, loss_g))
                loss_d_array.append(loss_d)
                loss_g_array.append(loss_g)
                #print([v.eval() for v in tf.all_variables() if v.name == "G/G/W1:0"])
                #print([v.eval() for v in tf.all_variables() if v.name == "D/D/W1:0"])
                self.anim_frames.append(self._samples(session))
    
        #session.close()
        self._plot_distributions(session)
        self._plot_loss(loss_d_array, loss_g_array)
        
        self._save_animation()
    def _samples(self, session, num_points=10000, num_bins=100):
        '''
        Return a tuple (db, pd, pg), where db is the current decision
        boundary, pd is a histogram of samples from the data distribution,
        and pg is a histogram of generated samples.
        '''
        xs = np.linspace(-self.noise_range, self.noise_range, num_points)
        bins = np.linspace(-self.noise_range, self.noise_range, num_bins)

        # decision boundary
        db = np.zeros((num_points, 1))
        for i in range(num_points // self.batch_size):
            db[self.batch_size * i:self.batch_size * (i + 1)] = self.D1.eval(feed_dict = {
                self.x: np.reshape(
                    xs[self.batch_size * i:self.batch_size * (i + 1)],
                    (self.batch_size, 1)
                )
            })

        # data distribution
        d = self.data_samples(num_points)
        pd, _ = np.histogram(d, bins=bins, density=True)

        # generated samples
        zs = self.noise_samples(num_points)#np.linspace(-self.noise_range, self.noise_range, num_points)
        g = np.zeros((num_points, 1))
        for i in range(num_points // self.batch_size):
            g[self.batch_size * i:self.batch_size * (i + 1)] = self.G.eval(feed_dict = {
                self.z: np.reshape(
                    zs[self.batch_size * i:self.batch_size * (i + 1)],
                    (self.batch_size, 1)
                )
            })
        pg, _ = np.histogram(g, bins=bins, density=True)

        return db, pd, pg

    def _plot_distributions(self, session):
        db, pd, pg = self._samples(session)
        print len(pd)
        print self.noise_range
        db_x = np.linspace(-self.noise_range, self.noise_range, len(db))
        p_x = np.linspace(-self.noise_range, self.noise_range, len(pd))
        f, ax = plt.subplots(1)
        ax.plot(db_x, db, label='decision boundary')
        ax.set_ylim(0, np.max(pg))
        plt.plot(p_x, pd, label='real data')
        plt.plot(p_x, pg, label='generated data')
        plt.title("1D GAN Data LR = %f, Gen LR = %f"%(self.data_learning_rate, self.gen_learning_rate))
        plt.xlabel('Data values')
        plt.ylabel('Probability density')
        plt.legend()
        plt.show()
        plt.savefig("dist_NR%f_M%f_STD%f_DLR%f_GLR%f.png"%(self.noise_range, self.data_mu, self.data_std, self.data_learning_rate, self.gen_learning_rate))
    def _plot_loss(self, loss_d_array, loss_g_array):
        xx = np.linspace(1,len(loss_d_array))*20
        plt.figure()
        plt.plot(loss_d_array)
        plt.plot(loss_g_array)
        plt.legend(['Discriminator', 'Generator'])
        plt.title("Training loss with Data LR = %f, Gen LR = %f"%(self.data_learning_rate, self.gen_learning_rate))
        plt.show()
        plt.savefig("loss_NR%f_M%f_STD%f_DLR%f_GLR%f.png"%(self.noise_range, self.data_mu, self.data_std, self.data_learning_rate, self.gen_learning_rate))
        
    def _save_animation(self):
        f, ax = plt.subplots(figsize=(6, 4))
        f.suptitle("1D GAN Data LR = %f, Gen LR = %f"%(self.data_learning_rate, self.gen_learning_rate))
        plt.xlabel('Data values')
        plt.ylabel('Probability density')
        ax.set_xlim(-6, 6)
        ax.set_ylim(0, 1.4)
        line_db, = ax.plot([], [], label='decision boundary')
        line_pd, = ax.plot([], [], label='real data')
        line_pg, = ax.plot([], [], label='generated data')
        frame_number = ax.text(
            0.02,
            0.95,
            '',
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes
        )
        ax.legend()

        db, pd, _ = self.anim_frames[0]
        db_x = np.linspace(-self.noise_range, self.noise_range, len(db))
        p_x = np.linspace(-self.noise_range, self.noise_range, len(pd))

        def init():
            line_db.set_data([], [])
            line_pd.set_data([], [])
            line_pg.set_data([], [])
            frame_number.set_text('')
            return (line_db, line_pd, line_pg, frame_number)

        def animate(i):
            frame_number.set_text(
                'Frame: {}/{}'.format(i, len(self.anim_frames))
            )
            db, pd, pg = self.anim_frames[i]
            line_db.set_data(db_x, db)
            line_pd.set_data(p_x, pd)
            line_pg.set_data(p_x, pg)
            return (line_db, line_pd, line_pg, frame_number)

        anim = animation.FuncAnimation(
            f,
            animate,
            init_func=init,
            frames=len(self.anim_frames),
            blit=True
        )
        anim.save("anim_NR%f_M%f_STD%f_DLR%f_GLR%f.gif"%(self.noise_range, self.data_mu, self.data_std, self.data_learning_rate, self.gen_learning_rate), fps=15, extra_args=['-vcodec', 'libx264'])

def optimizer(learning_rate, loss, var_list):
    initial_learning_rate = learning_rate
    decay = 0.95
    num_decay_steps = 100
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(initial_learning_rate,batch,num_decay_steps,decay,staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=batch,var_list=var_list)
    return optimizer
    

main()
