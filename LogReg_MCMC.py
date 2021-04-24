import numpy as np
import random
import math
import matplotlib.pyplot as plt
from math import exp
import os

random.seed()


class LOG_REG:

    def __init__(self, num_epocs, train_data, test_data, num_features, learn_rate, activation):
        self.train_data = train_data
        self.test_data = test_data
        self.num_features = num_features
        self.num_outputs = self.train_data.shape[1] - num_features
        self.num_train = self.train_data.shape[0]
        self.w = np.random.uniform(-0.5, 0.5, num_features)
        self.b = np.random.uniform(-0.5, 0.5, self.num_outputs)
        self.learn_rate = learn_rate
        self.max_epoch = num_epocs
        self.use_activation = activation
        self.out_delta = np.zeros(self.num_outputs)
        self.activation = activation

    def ACT(self, z_vec):
        if self.use_activation == False:
            y = 1 / (1 + np.exp(-1 * z_vec))
        else:
            y = z_vec
        return y

    def predict(self, x_vec):
        z_vec = x_vec.dot(self.w) - self.b
        output = self.ACT(z_vec)
        return output

    def squared_error(self, prediction, actual):
        return np.sum(np.square(prediction - actual)) / prediction.shape[0]

    def encode(self, w):
        self.w = w[0:self.num_features]
        self.b = w[self.num_features]

    def EVAL(self, data, w):
        self.encode(w)
        xx = np.zeros(data.shape[0])

        for s in range(0, data.shape[0]):
            i = s
            input_instance = data[i, 0:self.num_features]
            actual = data[i, self.num_features:]
            prediction = self.predict(input_instance)
            xx[s] = prediction
            
            # WE SHOULD ADD SOMETHING MORE HERE IF WE NEED IT
            
        return xx
    
    # ADD GRADIENT DESCENT FUNCTION HERE TO MAKE LOGISTIC REGRESSOR WORK WITHOUT MCMC



class MCMC:
    def __init__(self, samples, train_data, test_data, n_features, regression):
        # NUMBER OF SAMPLES DRAWN
        random.seed()
        self.samples = samples
        self.n_features = n_features
        self.train_data = train_data
        self.test_data = test_data
        self.regression = regression
        
    def MSE(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    def Likelihood(self, model, data, w, tau_sq):
        y = data[:, self.n_features[0]]
        xx = model.EVAL(data, w)
        accuracy = self.MSE(xx, y)
        loss = -0.5 * np.log(2 * math.pi * tau_sq) - 0.5 * np.square(y - xx) / tau_sq
        return [np.sum(loss), xx, accuracy]

    def prior_likelihood(self, sigma_squared, nu_1, nu_2, w, tau_sq):
        param = self.n_features[0] + 1
        part1 = -1 * (param / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
        log_loss = part1 - part2 - (1 + nu_1) * np.log(tau_sq) - (nu_2 / tau_sq)
        return log_loss
    
    def SAMPL(self):
        testsize = self.test_data.shape[0]
        trainsize = self.train_data.shape[0]
        samples = self.samples

        x_test = np.linspace(0, 1, num=testsize)
        x_train = np.linspace(0, 1, num=trainsize)

        y_test = self.test_data[:, self.n_features[0]]
        y_train = self.train_data[:, self.n_features[0]]
        w_size = self.n_features[0] + self.n_features[1]
        pos_w = np.ones((samples, w_size))
        pos_tau = np.ones((samples, 1))

        xxtrain_samples = np.ones((samples, trainsize))
        xxtest_samples = np.ones((samples, testsize))

        rmse_train = np.zeros(samples)
        rmse_test = np.zeros(samples)

        w = np.random.randn(w_size)
        w_proposal = np.random.randn(w_size)
        
        
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # 
        # PLEASE DO NOT TOUCH THESE VALUES UNLESS YOU KNOW WHAT YOU ARE DOING
        step_w = 0.05
        step_eta = 0.01
        # PLEASE DO NOT TOUCH THESE VALUES UNLESS YOU KNOW WHAT YOU ARE DOING
        # 
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        model = LOG_REG(0, self.train_data, self.test_data, self.n_features[0], 0.1, self.regression)

        pred_train = model.EVAL(self.train_data, w)
        
        # Use this if you want to
        pred_test = model.EVAL(self.test_data, w)

        eta = np.log(np.var(pred_train - y_train))
        tau_pro = np.exp(eta)

        print('evaluate Initial w')
        # THESE PARAMETERS REMAINS TO BE OPTIMIZED
        sigma_squared = 5
        nu_1 = 0
        nu_2 = 0

        prior_likelihood = self.prior_likelihood(sigma_squared, nu_1, nu_2, w, tau_pro)
        [likelihood, pred_train, rmsetrain] = self.Likelihood(model, self.train_data, w, tau_pro)
        # print(likelihood, ' initial likelihood')
        [likelihood_ignore, pred_test, rmsetest] = self.Likelihood(model, self.test_data, w, tau_pro)

        num_accept = 0

        for i in range(samples - 1):

            w_proposal = w + np.random.normal(0, step_w, w_size)
            eta_pro = eta + np.random.normal(0, step_eta, 1)
            tau_pro = math.exp(eta_pro)
            [likelihood_proposal, pred_train, rmsetrain] = self.Likelihood(model, self.train_data, w_proposal,tau_pro)
            [likelihood_ignore, pred_test, rmsetest] = self.Likelihood(model, self.test_data, w_proposal, tau_pro)

            prior_prop = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal,tau_pro)
            diff_likelihood = likelihood_proposal - likelihood
            diff_priors = prior_prop - prior_likelihood
            Probs = min(1, int(math.exp(diff_likelihood + diff_priors)))
            u = random.uniform(0, 1)
            if u < Probs:
                num_accept += 1
                likelihood = likelihood_proposal
                prior_likelihood = prior_prop
                w = w_proposal
                eta = eta_pro
                rmse_train[i + 1,] = rmsetrain
                rmse_test[i + 1,] = rmsetest
                pos_w[i + 1,] = w_proposal
                pos_tau[i + 1,] = tau_pro
                xxtrain_samples[i + 1,] = pred_train
                xxtest_samples[i + 1,] = pred_test

            else:
                pos_w[i + 1,] = pos_w[i,]
                pos_tau[i + 1,] = pos_tau[i,]
                xxtrain_samples[i + 1,] = xxtrain_samples[i,]
                xxtest_samples[i + 1,] = xxtest_samples[i,]
                rmse_train[i + 1,] = rmse_train[i,]
                rmse_test[i + 1,] = rmse_test[i,]

        accept_ratio = num_accept / (samples * 1.0) * 100

        print(accept_ratio, '% was accepted')

        # BURN IN TIME
        burnin = 0.35 * samples

        pos_w = pos_w[int(burnin):, ]
        pos_tau = pos_tau[int(burnin):, ]
        rmse_train = rmse_train[int(burnin):]
        rmse_test = rmse_test[int(burnin):]

        rmse_tr = np.mean(rmse_train)
        rmsetr_std = np.std(rmse_train)
        rmse_tes = np.mean(rmse_test)
        rmsetest_std = np.std(rmse_test)
        
        # print(rmse_tr, rmsetr_std, rmse_tes, rmsetest_std, ' rmse_tr, rmsetr_std, rmse_tes, rmsetest_std')

        num_trials = 100
        accuracy = np.zeros(num_trials)

        for i in range(num_trials):
            # print(pos_w.mean(axis=0), pos_w.std(axis=0), ' pos w mean, pos w std')
            w_drawn = np.random.normal(pos_w.mean(axis=0), pos_w.std(axis=0), w_size)
            tausq_drawn = np.random.normal(pos_tau.mean(),
                                           pos_tau.std())  # a buf is present here - gives negative values at times

            [loss, xx_, accuracy[i]] = self.Likelihood(model, self.test_data, w_drawn, tausq_drawn)

            # PRINT OF THE POSTERIOR VALUES
            # print(i, loss, accuracy[i], tausq_drawn, pos_tau.mean(), pos_tau.std(), ' posterior value')

        print(accuracy.mean(), accuracy.std(), ' is mean and std of accuracy MSE test')

        return (pos_w, pos_tau, xxtrain_samples, xxtest_samples, rmse_train, rmse_test, accept_ratio)


def histogram_trace(pos_points, fname):
    # CHANGE IF YOU DO NOT LIKE HOW THEY LOOK LIKE. 
    # ALSO, DISCUSS IF WE ACTUALLY NEED TO PRESENT THIS PART OF THE ALGORITHM
    size = 15
    plt.tick_params(labelsize=size)
    params = {'legend.fontsize': size, 'legend.handlelength': 2}
    plt.rcParams.update(params)
    plt.grid(alpha=0.75)

    plt.hist(pos_points, bins=20, color='#0504aa', alpha=0.7)
    plt.title("POS distribution ", fontsize=size)
    plt.xlabel(' Parameter value  ', fontsize=size)
    plt.ylabel(' Frequency ', fontsize=size)
    plt.tight_layout()
    plt.savefig(fname + '_posterior.png')
    plt.clf()

    plt.tick_params(labelsize=size)
    params = {'legend.fontsize': size, 'legend.handlelength': 2}
    plt.rcParams.update(params)
    plt.grid(alpha=0.75)
    plt.plot(pos_points)

    plt.title("Parameter trace plot", fontsize=size)
    plt.xlabel(' Number of Samples  ', fontsize=size)
    plt.ylabel(' Parameter value ', fontsize=size)
    plt.tight_layout()
    plt.savefig(fname + '_trace.png')
    plt.clf()


def main():
    activation = True

    # import numpy as np
    import pandas as pd

    # THIS SECTION IF FOR THE OTHER ORIGINAL DATASET.
    # df = pd.read_csv('Parsed_Data/California.csv', sep=',', header=None)
    # print(df)
    # df=df.dropna(axis=1)
    # df1 = (df.iloc[73:377])
    # # df1 = np.delete(df1,[3,4,5,6,7,8],axis=1)
    # df1 = df1.drop([3, 4, 5, 6, 7, 8], axis=1)
    # # df2 = (df.iloc[378:419])
    # # print(df2)
    # # print(df.iloc[80])
    # # y = df1[1]
    # df1 = df1.drop([0], axis=1)
    # df1 = df1.reset_index(drop=True)

    # NOTE: YOU NEED TO DROP EMPTY COLUMNS OF DATA
    # I USE THESE ARRAYS FOR THAT PURPOSE.
    # REMEMBER TO DELETE NULL COLUMNS
    dd = pd.read_csv('owid-covid-data.csv', sep=',', header=None, low_memory=False)
    dd = dd.iloc[36:418]
    arr1 = [0, 1, 2, 3, 50, 53, 54]
    arr2 = np.arange(16, 52)
    arr = np.hstack((arr1, arr2))
    dd = dd.drop(arr, axis=1)
    dd = dd.reset_index(drop=True)

    # UNCOMMENT FOR THE ORIGINAL DATA SECTION
    # train_data =np.asarray(df1.values[:250,:]).astype(np.float)
    # test_data = np.asarray(df1.values[200:,:]).astype(np.float)
    # features = 5

    train_data = np.asarray(dd.values[:250, :]).astype(np.float)
    test_data = np.asarray(dd.values[250:, :]).astype(np.float)

    # SET THE NUMBER OF FEATURES AND OUTPUTS.
    # THE DATASET THAT I FOUND OUT GIVES US 24 USABLE FEATURES.
    # THE DATASET THAT WE PLAN TO USE DOES NOT HAVE THIS MANY. SO PLEASE ADJUST ACCORDINGLY.
    # features = 23

    features = 15
    output = 1

    # idx = np.arange(4,24)
    idx = np.arange(4, 16)
    idx = np.hstack((idx, [0, 1, 2, 3]))

    # UNCOMMENT THIS IF WE ARE USING THE ORIGINAL DATASET
    # permutation = [0, 4, 3, 2, 1]
    # idx = np.empty_like(permutation)
    # idx[permutation] = np.arange(len(permutation))
    # idx = [0, 5, 4, 3, 2, 1]

    train_data = train_data[:, idx]
    test_data = test_data[:, idx]
    print(train_data)

    # SAVE THE DATA TO USE FOR OUR RNN
    import scipy.io as sio
    sio.savemat('data.mat', {'train': train_data, 'test': test_data}, do_compression=True)

    n_features = [features, output]

    MinCriteria = 0.005
    numSamples = 5000

    mcmc = MCMC(numSamples, train_data, test_data, n_features, activation)

    [pos_w, pos_tau, xx_train, xx_test, rmse_train, rmse_test, accept_ratio] = mcmc.SAMPL()
    print('MCMC DONE !!')

    # OPTIONAL SECTION FOR GETTING POSTERIORS. MIGHT BE RELEVANT TO LOOK AT
    # folder = 'POS'
    # if not os.path.exists(folder):
    #     os.makedirs(folder)
    # for i in range(pos_w.shape[1]):
    #     histogram_trace(pos_w[:, i], folder + '/' + str(i))
    # mpl_fig = plt.figure()
    # ax = mpl_fig.add_subplot(111)
    #
    # ax.boxplot(pos_w)
    # ax.set_xlabel('[w0] [w1] [w3] [b]')
    # ax.set_ylabel('POS')
    #
    # plt.title("POS")
    # plt.savefig('w_pos.png')
    # plt.clf()

    xx_mu = xx_test.mean(axis=0)
    xx_high = np.percentile(xx_test, 60, axis=0)
    xx_low = np.percentile(xx_test, 20, axis=0)

    xx_mu_tr = xx_train.mean(axis=0)
    xx_high_tr = np.percentile(xx_train, 60, axis=0)
    xx_low_tr = np.percentile(xx_train, 20, axis=0)

    ytest_data = test_data[:, features]
    ytrain_data = train_data[:, features]

    x_test = np.linspace(0, 1, num=test_data.shape[0])
    x_train = np.linspace(0, 1, num=train_data.shape[0])

    plt.plot(x_test, ytest_data, label='actual')
    plt.plot(x_test, xx_mu, label='mean prediction')
    plt.plot(x_test, xx_low, label='pred: 20th percentile')
    plt.plot(x_test, xx_high, label='pred: 60th percentile')
    plt.fill_between(x_test, xx_low, xx_high, facecolor='g', alpha=0.4)
    plt.legend(loc='upper right')
    plt.title("Test Data")
    plt.savefig('test.png')
    plt.clf()

    plt.plot(x_train, ytrain_data, label='actual')
    plt.plot(x_train, xx_mu_tr, label='mean prediction')
    plt.plot(x_train, xx_low_tr, label='pred: 20th percentile')
    plt.plot(x_train, xx_high_tr, label='pred: 60th percentile')
    plt.fill_between(x_train, xx_low_tr, xx_high_tr, facecolor='g', alpha=0.4)
    plt.legend(loc='upper right')
    plt.title("Train Data")
    plt.savefig('train.png')
    plt.clf()




if __name__ == "__main__":
    main()
