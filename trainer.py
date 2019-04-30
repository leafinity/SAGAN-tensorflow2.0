import time
import glob
import tensorflow as tf
import numpy as np
from sagan_models import create_generator, create_discriminator

class Trainer(object):
    
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.epoch = config.epoch
        self.batch_size = config.batch_size
        self.print_freq = config.print_freq
        self.save_freq = config.save_freq
        self.gpl = config.gpl
        self.sample_num = config.sample_num
        self.image_path = config.image_path
        self.checkpoint_dir = config.checkpoint_dir
        self.result_dir = config.result_dir
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir

        # initial models
        self.g = create_generator(
            image_size=config.img_size,
            z_dim=config.z_dim,
            filters=config.g_conv_filters,
            kernel_size=config.g_conv_kernel_size)

        self.d = create_discriminator(
            image_size=config.img_size,
            filters=config.d_conv_filters,
            kernel_size=config.d_conv_kernel_size)

        # initial optimizers
        self.g_opt = tf.optimizers.get(config.g_opt)
        self.g_opt.learning_rate = config.g_lr
        if isinstance(self.g_opt, tf.optimizers.Adam):
            self.g_opt.beta_1=config.beta1
            self.g_opt.beta_2=config.beta2

        self.d_opt = tf.optimizers.get(config.d_opt)
        self.d_opt.learning_rate = config.d_lr
        if isinstance(self.g_opt, tf.optimizers.Adam):
            self.d_opt.beta_1=config.beta1
            self.d_opt.beta_2=config.beta2

        if config.load_model:
            self.load_model()

    @staticmethod
    def w_loss(y_true, y_pred):
        return tf.reduce_mean(y_true * y_pred)

    def data_generator(self):
        images = glob.glob(config.image_path)
        self.nbatch = int(np.ceil(len(images)/self.batch_size))

    def gradient_penalty(self, real, fake):
        alpha = tf.random.uniform(shape=[len(real), 1, 1, 1], minval=0., maxval=1.)
        interpolated = alpha * real + (1 - alpha) * fake
    
        with tf.GradientTape() as tape_p:
            tape_p.watch(interpolated)
            logit = self.d(interpolated)
        
        grad = tape_p.gradient(logit, interpolated)
        grad_norm = tf.norm(tf.reshape(grad, (batch_size, -1)), axis=1)

        return self.gpl * tf.reduce_mean(tf.square(grad_norm - 1.))

    def load_models(self):
        pass

    def save_models(self):
        pass

    def save_samples(self):
        pass

    def train_discriminator_step(self, real_img, noise_z):
        with tf.GradientTape() as tape_d:

            fake_img  = self.g(noise_z, training=False)

            real_pred = self.d(real_img, training=True)
            fake_pred = self.d(fake_img, training=True)
            
            y_true = tf.ones(shape=tf.shape(real_pred), dtype=tf.float32)

            real_loss = self.w_loss(-y_true, real)
            fake_loss = self.w_loss(sy_true, fake)

            gp = gradient_penalty(real_img, fake_img)
            
            total_loss = real_loss + fake_loss + gp

        gradients = tape_d.gradient(total_loss, D.trainable_variables)
        self.d_opt.apply_gradients(zip(gradients, D.trainable_variables))
        
        return total_loss, gp

    def train_generator_step(self, noise_z):
        with tf.GradientTape() as tape_g:
            fake_img  = G(noise_z, training=True)
            fake_pred = D(fake_img, training=False)

            g_loss = self.w_loss(fake_pred, -tf.ones(shape=tf.shape(real), dtype=tf.float32))

            gradients = tape_g.gradient(g_loss, G.trainable_variables)
            self.g_opt.apply_gradients(zip(gradients, G.trainable_variables))
        
        return g_loss

    def train(self):
        print("Start Training")
        print('epoch: {}'.format(self.epoch))
        
        for epoch in range(self.epoch):
            epoch_start = time.time()

            for i in range(self.nbatch):
                z = tf.random.truncated_normal(shape=(self.batch_size, self.z_dim), dtype=tf.float32)
                d_loss, gp_loss = train_discriminator_step(next(self.data_generator), z)
                g_loss = train_generator_step(z)

            if (epoch % self.print_freq) == 0:
                print('epoch {}/{} ({:.2f} sec):, d_loss {:.4f}, gp_loss {:.4f}, g_loss {:.4f}'.format([
                    epoch, self.epoch, 
                    time.time() - epoch_start,
                    d_loss.numpy(), gp_loss.numpy(), g_loss.numpy()]))

            if (epoch % self.save_freq):
                self.save_models()

                # z = tf.random.truncated_normal(shape=(self.sample_num, self.z_dim), dtype=tf.float32)
                # self.save_samples(self.g(z))

    def test(self):
        z = tf.random.truncated_normal(shape=(self.sample_num, self.z_dim), dtype=tf.float32)
        # self.save_samples(self.g(z), path=self.result_dir)
        pass



