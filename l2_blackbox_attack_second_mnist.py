
import sys
import tensorflow as tf
import numpy as np
from numpy import linalg as LA
from scipy import io

BINARY_SEARCH_STEPS = 30  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 10000  # number of iterations to perform gradient descent
ABORT_EARLY = True  # if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-2  # larger values converge faster to less accurate results
TARGETED = False  # should we target one specific class? or just be wrong?
CONFIDENCE = 0  # how strong the adversarial example should be
INITIAL_CONST = 1e-3  # the initial constant c to pick as a first guess
RO = 20
u = 1
Query_iterations = 40
GAMA = 1
EPI = 0.2
ALPHA=10


class LADMMBB:
    def __init__(self, sess, model, batch_size=1, confidence=CONFIDENCE,
                 targeted=TARGETED, learning_rate=LEARNING_RATE,
                 binary_search_steps=BINARY_SEARCH_STEPS, max_iterations=MAX_ITERATIONS,
                 abort_early=ABORT_EARLY, ro=RO, gama=GAMA, epi=EPI, alpha=ALPHA):
        """
        The L_2 optimized attack.
        """

        self.model = model
        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.batch_size = batch_size
        self.ro = ro
        self.grad = self.gradient_descent(sess, model)
        self.grad2 = self.gradient_descent2(sess, model)
        self.grad3 = self.gradient_descent3(sess, model)
        self.gama = gama
        self.epi=epi
        self.alpha = alpha

    def compare(self, x, y):
        if not isinstance(x, (float, int, np.int64)):
            x = np.copy(x)
            if self.TARGETED:
                x[y] -= self.CONFIDENCE
            else:
                x[y] += self.CONFIDENCE
            x = np.argmax(x)
        if self.TARGETED:
            return x == y
        else:
            return x != y

    def gradient_descent(self, sess, model):

        batch_size = self.batch_size
        shape = (batch_size, model.image_size, model.image_size, model.num_channels)

        tz = tf.Variable(np.zeros(shape, dtype=np.float32))
        timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        tlab = tf.Variable(np.zeros((batch_size, model.num_labels)), dtype=tf.float32)

        # and here's what we use to assign them
        assign_timg = tf.placeholder(tf.float32, shape)
        assign_tlab = tf.placeholder(tf.float32, (batch_size, model.num_labels))
        assign_tz = tf.placeholder(tf.float32, shape)

        # the resulting image, tanh'd to keep bounded from boxmin to boxmax
        newimg = tz + timg
        l2dist_real = tf.reduce_sum(tf.square(tz), [1, 2, 3])
        output = model.predict(newimg)

        real = tf.reduce_sum(tlab * output, 1)
        other = tf.reduce_max((1 - tlab) * output - (tlab * 10000), 1)

        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            loss1 = tf.maximum(0.0, other - real + self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(0.0, real - other + self.CONFIDENCE)

 #       loss1 = tf.reduce_sum(loss1)

        # these are the variables to initialize when we run
        setup = []
        setup.append(timg.assign(assign_timg))
        setup.append(tlab.assign(assign_tlab))
        setup.append(tz.assign(assign_tz))

        def doit(imgs, labs, z):

            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]

            sess.run(setup, {assign_timg: batch, assign_tlab: batchlab, assign_tz: z,})

            l2s, scores, nimg, loss = sess.run([l2dist_real, output, newimg, loss1])

            return l2s, scores, nimg, loss

        return doit

    def gradient_descent2(self, sess, model):

        batch_size = self.batch_size
        shape = (batch_size, model.image_size, model.image_size, model.num_channels)

        tz = tf.Variable(np.zeros(shape, dtype=np.float32))
        timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        tlab = tf.Variable(np.zeros((batch_size, model.num_labels)), dtype=tf.float32)

        # and here's what we use to assign them
        assign_timg = tf.placeholder(tf.float32, shape)
        assign_tlab = tf.placeholder(tf.float32, (batch_size, model.num_labels))
        assign_tz = tf.placeholder(tf.float32, shape)

        # the resulting image, tanh'd to keep bounded from boxmin to boxmax
        newimg = tz + timg
        l2dist_real = tf.reduce_sum(tf.square(tz), [1, 2, 3])
        output = model.predict(newimg)

        real = tf.reduce_sum(tlab * output, 1)
        other = tf.reduce_max((1 - tlab) * output - (tlab * 10000), 1)

        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            loss1 = tf.maximum(0.0, other - real + self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(0.0, real - other + self.CONFIDENCE)

        # these are the variables to initialize when we run
        setup = []
        setup.append(timg.assign(assign_timg))
        setup.append(tlab.assign(assign_tlab))
        setup.append(tz.assign(assign_tz))

        def doit(imgs, labs, z):

            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]

            sess.run(setup, {assign_timg: batch, assign_tlab: batchlab, assign_tz: z,})

            loss = sess.run([loss1])

            return loss

        return doit

    def gradient_descent3(self, sess, model):

        batch_size = self.batch_size
        shape = (batch_size, model.image_size, model.image_size, model.num_channels)

        tz = tf.Variable(np.zeros(shape, dtype=np.float32))
        timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        tlab = tf.Variable(np.zeros((batch_size, model.num_labels)), dtype=tf.float32)

        # and here's what we use to assign them
        assign_timg = tf.placeholder(tf.float32, shape)
        assign_tlab = tf.placeholder(tf.float32, (batch_size, model.num_labels))
        assign_tz = tf.placeholder(tf.float32, shape)

        # the resulting image, tanh'd to keep bounded from boxmin to boxmax
        newimg = tz + timg
        l2dist_real = tf.reduce_sum(tf.square(tz), [1, 2, 3])
        output = model.predict(newimg)

        real = tf.reduce_sum(tlab * output, 1)
        other = tf.reduce_max((1 - tlab) * output - (tlab * 10000), 1)

        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            loss1 = tf.maximum(0.0, other - real + self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(0.0, real - other + self.CONFIDENCE)


        loss1 =  tf.reduce_sum(loss1)
        #loss1 = loss1*loss1
        gradtz = tf.gradients(loss1, [tz])

        probabi = tf.nn.softmax(output)
        real_pro = tf.reduce_sum(tlab * probabi)
        other_pro = 1.0001-real_pro
        loss2 = tf.log(real_pro )
        loss3 = tf.log(other_pro )
#        loss2 = tf.reduce_sum(real)
#        loss3 = tf.reduce_sum(output)
        # these are the variables to initialize when we run

        grad_loglike = tf.gradients(loss2, [tz])
        setup = []
        setup.append(timg.assign(assign_timg))
        setup.append(tlab.assign(assign_tlab))
        setup.append(tz.assign(assign_tz))

        def doit(imgs, labs, z):

            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]

            sess.run(setup, {assign_timg: batch, assign_tlab: batchlab, assign_tz: z,})

            loss, value_log , lo3 , realpro, gradz, grad_log= sess.run([loss1, loss2, loss3, real_pro,gradtz, grad_loglike])

            return loss, value_log, lo3, realpro, gradz, grad_log

        return doit

    def attack(self, imgs, targets):
        """
        Perform the L_2 attack on the given images for the given targets.
        """
        r = []
        qc = []
        ql2 = []
        print('go up to', len(imgs))
        for i in range(0, len(imgs), self.batch_size):
            print('tick', i)
            attac, queryc, queryl2 = self.attack_batch(imgs[i:i + self.batch_size], targets[i:i + self.batch_size])
            r.extend(attac)
            qc.extend(queryc)
            ql2.extend(queryl2)
        return np.array(r), np.array(qc), np.array(ql2)

    def attack_batch(self, imgs, labs):
        """
        Run the attack on a batch of images and labels.
        """
        batch_size = self.batch_size
        o_bestl2 = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestattack = [imgs[0]] * batch_size
        o_querycount = [1e10] * batch_size
        o_queryl2 = [1e10] * batch_size
        o_conv = [1e10] * self.BINARY_SEARCH_STEPS

        delt = 0.0 * np.ones(imgs.shape)
        s = 0.0 * np.ones(imgs.shape)

        lr = 0.0004

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            print(outer_step, o_bestl2)

     #       aa,bb,cc,dd = self.grad(imgs, labs, delt)
            loss, log_value , log_value_other, realpro ,_,_  = self.grad3(imgs, labs, delt)

            delt_grads = np.zeros(imgs.shape)

            grad_log = np.zeros(imgs.shape)
            grads_log_other = np.zeros(imgs.shape)
            for iii in range(Query_iterations):
                dirc = np.random.normal(0, 1, imgs.shape)
                sss = LA.norm(dirc.reshape(batch_size,-1), axis=1)
                sss = sss.reshape(batch_size, 1, 1, 1)
                dir_normal = dirc / sss
                temploss, temp_log, temp_log_other, _,_,_  = self.grad3( imgs + u * dir_normal, labs, delt)
                delt_grads = delt_grads + (temploss - loss).reshape(batch_size, 1, 1, 1)*dir_normal

                grad_log = grad_log + (temp_log - log_value).reshape(batch_size, 1, 1, 1)*dir_normal

                grads_log_other = grads_log_other + (temp_log_other - log_value_other).reshape(batch_size, 1, 1, 1)*dir_normal

            delt_grads = delt_grads*imgs.shape[1]*imgs.shape[2]*imgs.shape[3]/(u * Query_iterations)

    #        delt_grads = delt_grads / (u * Query_iterations)

            grad_log = grad_log*imgs.shape[1]*imgs.shape[2]*imgs.shape[3]/(u * Query_iterations)
     #       grad_log = grad_log /(u * Query_iterations)
            grads_log_other = grads_log_other*imgs.shape[1]*imgs.shape[2]*imgs.shape[3]/(u * Query_iterations)

            vec_grad_loss = np.reshape(delt_grads, (-1, 1))

            vec_grads_log = np.reshape(grad_log, (-1, 1))
            vec_grads_log_T = np.transpose(vec_grads_log)

            ver_grads_other = np.reshape(grads_log_other, (-1, 1))
            ver_grads_other_T = np.transpose(ver_grads_other)

            FisherMat = realpro * np.matmul(vec_grads_log, vec_grads_log_T) + (1 - realpro) * np.matmul(ver_grads_other,
                                                                                               ver_grads_other_T)
     #       FisherMat = np.matmul(vec_grads_log, vec_grads_log_T)

            inv_fisher = np.linalg.inv(FisherMat + 0.000005 * np.eye(vec_grads_log.size))
  #          np.savetxt('invFisher.csv', inv_fisher, delimiter=',')
            update_grad = np.matmul(inv_fisher, vec_grad_loss)

            delt = delt - lr * np.reshape(update_grad, imgs.shape)

      #      delt = delt - lr * delt_grads

            mintemp = np.where( 0.5-imgs < self.epi,  0.5-imgs, self.epi)
            maxtemp = np.where( -0.5-imgs > -self.epi, -0.5-imgs, -self.epi)
            delttemp = np.where( delt > mintemp, mintemp, delt )
            delt = np.where( delttemp < maxtemp, maxtemp, delttemp )

            l2s, scores, nimg, loss1 = self.grad(imgs, labs, delt)
            print(loss1[0])

            for e, (l2, sc, ii,) in enumerate(zip(l2s, scores, nimg)):
                if l2 < o_bestl2[e] and self.compare(sc, np.argmax(labs[e])):
                    o_bestl2[e] = l2
                    o_bestscore[e] = np.argmax(sc)
                    o_bestattack[e] = ii
                    if o_querycount[e] == 1e10:
                        o_querycount[e] = (outer_step + 1) * ( Query_iterations + 1)
                        o_queryl2[e] = l2

            if np.max(o_querycount) < 1e10:
                print(outer_step, o_bestl2)
                break

        return o_bestattack, o_querycount, o_queryl2


    def attack_batch2(self, imgs, labs):
        """
        Run the attack on a batch of images and labels.
        """
        batch_size = self.batch_size
        o_bestl2 = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestattack = [imgs[0]] * batch_size
        o_querycount = [1e10] * batch_size
        o_queryl2 = [1e10] * batch_size
        o_conv = [1e10] * self.BINARY_SEARCH_STEPS

        delt = 0.0 * np.ones(imgs.shape)
        s = 0.0 * np.ones(imgs.shape)

        lr = 0.0001

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            print(outer_step, o_bestl2)

     #       aa,bb,cc,dd = self.grad(imgs, labs, delt)
            loss, log_value , log_value_other, realpro, gradz, grad_loghood  = self.grad3(imgs, labs, delt)
            print(loss)

            delt_grads = np.zeros(imgs.shape)

            grad_log = np.zeros(imgs.shape)
     #       grads_log_other = np.zeros(imgs.shape)
            for iii in range(Query_iterations):
                dirc = np.random.normal(0, 1, imgs.shape)
                sss = LA.norm(dirc.reshape(batch_size,-1), axis=1)
                sss = sss.reshape(batch_size, 1, 1, 1)
                dir_normal = dirc / sss
     #           dir_normal = dirc
                temploss, temp_log, temp_log_other,_,_ ,_ = self.grad3( imgs + u * dir_normal, labs, delt)
                delt_grads = delt_grads + (temploss - loss).reshape(batch_size, 1, 1, 1)*dir_normal

                grad_log = grad_log + (temp_log - log_value).reshape(batch_size, 1, 1, 1)*dir_normal

    #            grads_log_other = grads_log_other + (temp_log_other - log_value_other).reshape(batch_size, 1, 1, 1)*dir_normal

            delt_grads = delt_grads*imgs.shape[1]*imgs.shape[2]*imgs.shape[3]/(u * Query_iterations)

     #       delt_grads = delt_grads / (u * Query_iterations)


            grad_log = grad_log*imgs.shape[1]*imgs.shape[2]*imgs.shape[3]/(u * Query_iterations)
     #       grad_log = grad_log /(u * Query_iterations)

            vec_grad_loss = np.reshape(delt_grads, (-1, 1))

            vec_grads_log = np.reshape(grad_log, (-1, 1))
    #        vec_grads_log_T = np.transpose(vec_grads_log)

            sigm = np.linalg.norm(vec_grads_log)
            ucc = vec_grads_log / sigm
            ucc_T = np.reshape(ucc, [1, -1])
            lamb = 0.01

            update_grad = np.matmul(ucc_T, vec_grad_loss)
            update_grad = (1 / (sigm * sigm + lamb) - 1 / lamb) * np.matmul(ucc, update_grad) + 1 / lamb * vec_grad_loss

         #   FisherMat = realpro * np.matmul(vec_grads_log, vec_grads_log_T) + (1 - realpro) * np.matmul(ver_grads_other,
                                                                                           #     ver_grads_other_T)
     #       FisherMat = np.matmul(vec_grads_log, vec_grads_log_T)

   #         inv_fisher = np.linalg.inv(FisherMat + 0.001 * np.eye(vec_grads_log.size))

      #      update_grad = np.matmul(inv_fisher, vec_grad_loss)

            delt = delt - lr * np.reshape(update_grad, imgs.shape)

      #      delt = delt - lr * delt_grads

            mintemp = np.where( 0.5-imgs < self.epi,  0.5-imgs, self.epi)
            maxtemp = np.where( -0.5-imgs > -self.epi, -0.5-imgs, -self.epi)
            delttemp = np.where( delt > mintemp, mintemp, delt )
            delt = np.where( delttemp < maxtemp, maxtemp, delttemp )

            l2s, scores, nimg, loss1 = self.grad(imgs, labs, delt)
            print(loss1[0])

            for e, (l2, sc, ii,) in enumerate(zip(l2s, scores, nimg)):
                if l2 < o_bestl2[e] and self.compare(sc, np.argmax(labs[e])):
                    o_bestl2[e] = l2
                    o_bestscore[e] = np.argmax(sc)
                    o_bestattack[e] = ii
                    if o_querycount[e] == 1e10:
                        o_querycount[e] = (outer_step + 1) * ( Query_iterations + 1)
                        o_queryl2[e] = l2

            if np.max(o_querycount) < 1e10:
                print(outer_step, o_bestl2)
                break

        return o_bestattack, o_querycount, o_queryl2

    def attack_batch3(self, imgs, labs):
        """
        Run the attack on a batch of images and labels.
        """
        batch_size = self.batch_size
        o_bestl2 = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestattack = [imgs[0]] * batch_size
        o_querycount = [1e10] * batch_size
        o_queryl2 = [1e10] * batch_size
        o_conv = [1e10] * self.BINARY_SEARCH_STEPS

        delt = 0.0 * np.ones(imgs.shape)
        s = 0.0 * np.ones(imgs.shape)

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            print(outer_step, o_bestl2)

            loss = self.grad2(imgs, labs, delt)

            delt_grads = np.zeros(imgs.shape)

            for iii in range(Query_iterations):
                dirc = np.random.normal(0, 1, imgs.shape)
                sss = LA.norm(dirc.reshape(batch_size,-1), axis=1)
                sss = sss.reshape(batch_size, 1, 1, 1)
                dir_normal = dirc / sss
                temploss = self.grad2( imgs + u * dir_normal, labs, delt)
                delt_grads = delt_grads + (temploss[0] - loss[0]).reshape(batch_size, 1, 1, 1)*dir_normal

            delt_grads = delt_grads*imgs.shape[1]*imgs.shape[2]*imgs.shape[3]/(u * Query_iterations)

            lr = 0.01

            delt = delt - lr * delt_grads

            mintemp = np.where( 0.5-imgs < self.epi,  0.5-imgs, self.epi)
            maxtemp = np.where( -0.5-imgs > -self.epi, -0.5-imgs, -self.epi)
            delttemp = np.where( delt > mintemp, mintemp, delt )
            delt = np.where( delttemp < maxtemp, maxtemp, delttemp )

            l2s, scores, nimg, loss1 = self.grad(imgs, labs, delt)
            print(loss1[0])

            for e, (l2, sc, ii,) in enumerate(zip(l2s, scores, nimg)):
                if l2 < o_bestl2[e] and self.compare(sc, np.argmax(labs[e])):
                    o_bestl2[e] = l2
                    o_bestscore[e] = np.argmax(sc)
                    o_bestattack[e] = ii
                    if o_querycount[e] == 1e10:
                        o_querycount[e] = (outer_step + 1) * ( Query_iterations + 1)
                        o_queryl2[e] = l2

            if np.max(o_querycount) < 1e10:
                print(outer_step, o_bestl2)
                break


        return o_bestattack, o_querycount, o_queryl2

