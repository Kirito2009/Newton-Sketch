            #####random part#########
import numpy as np
import os
os.chdir(r'F:\random IPM\dataset')
from sklearn.datasets import load_svmlight_file
import os
from scipy.linalg import cholesky
import numpy as np
from numpy.linalg import solve
from scipy.sparse import coo_matrix, block_diag, eye, dia_matrix, csc_matrix, hstack, vstack
from scipy.linalg import ldl
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm
from numpy.linalg import inv as dense_inv
from scipy.sparse.linalg import inv
from scipy.sparse import diags



class SVM():
    def __init__(self, tao):
        os.chdir(r'F:\random IPM')
        os.chdir('.\dataset')
        self.tao = tao
        pass
    def load_data(self, filename):

        data = load_svmlight_file(filename)
        self.X = data[0]
        self.Y = data[1]
        self.Y[self.Y == 0] = -1
        print('data loaded')
        return
    def data_preprocess(self):
        
        self.X = self.X.T
        self.n = self.X.shape[1]
        self.m = self.X.shape[0]
        assert self.m<self.n
       
        Y = csc_matrix(self.Y)
        Y = diags(Y.A.ravel()).tocsc()
        self.XY = self.X.dot(Y)
        # del self.X
        print('XY computed')
        Q_w = np.ones(self.m)
        Q_w = csc_matrix(Q_w)
        self.Q_w = diags(Q_w.A.ravel()).tocsc()
        print('Q_w prepared')
        assert self.Q_w.shape == (self.m, self.m)
        del Q_w
        self.Q_z = csc_matrix((self.n, self.n))
        print('Q_z prepared')
        # 需要改成稀疏矩阵形式存储
        self.A_w = vstack((block_diag(np.ones(self.m)), csc_matrix((1, self.m))))
        self.A_z = vstack((-self.XY, self.Y.T))
        assert self.A_w.shape == (self.m + 1, self.m)
        assert self.A_z.shape == (self.m + 1, self.n)
        print('A_w,A_z prepared')

        self.c_z = np.ones(self.n)
        self.c_w = np.zeros(self.m)
        print('c_w,c_z prepared')

        self.b = np.zeros(self.m + 1)
        print('b prepared')

        self.u = self.tao * np.ones(self.n)
        print('u prepared')
        
        return self.A_w, self.A_z, self.Q_w, self.Q_z, self.c_w, self.c_z, self.b, self.u
    def gen_sketch_matrix(self,m,n):
        S = ((np.random.randn(m, n) > 0) * 2 - 1) / m
        return S
    def fit(self, A_w, A_z, Q_w, Q_z, c_w, c_z, b, u,
            init=(),
            maxk=130, eta=0.99995, eps_r_b=1e-5, K=3,
            eps_r_cw=1e-5, eps_r_cz=1e-5, eps_mu=1e-5, delta=0.1, beta_min=0.1, beta_max=10, gamma=0.1,
            ):
        w, z, y, s, v = init
        print('start computing')
        k = 0
        assert w.shape == (self.m,)
        assert z.shape == (self.n,)
        assert s.shape == (self.n,)
        assert v.shape == (self.n,)
        assert y.shape == (self.m + 1,)
        assert A_w.shape == (self.m + 1, self.m)
        assert A_z.shape == (self.m + 1, self.n)
        assert Q_w.shape == (self.m, self.m)
        assert Q_z.shape == (self.n, self.n)
        assert c_w.shape == (self.m,)
        assert c_z.shape == (self.n,)
        assert b.shape == (self.m + 1,)
        assert u.shape == (self.n,)
        Theta_inv = csc_matrix(s / z + v / (u - z))
        Theta_inv = diags(Theta_inv.A.ravel()).tocsc()
        r_b = -(A_w.dot(w) + A_z.dot(z) - b)
        r_cw = -(-Q_w.dot(w) + A_w.T.dot(y) - c_w)
        r_sz = -z*s
        r_vz = -(u-z)*v
        r_cz = -A_z.T.dot(y) +c_z +Q_z.dot(z)-s+v
        r_cz_hat = r_cz-r_sz/z+r_vz/(u-z)
        mu = (s.dot(z) + (u - z).dot(v)) / (2 * self.n)
        temp_A_z = A_z.dot(diags(1/(Q_z+Theta_inv).diagonal().ravel()).tocsr())
        r_b_hat = r_b+A_w.dot(inv(Q_w)).dot(r_cw)+temp_A_z.dot(r_cz_hat)
        error_log = list()
        while k<= maxk and( norm(r_b)/(1+norm(self.b))>= eps_r_b or norm(r_cw)/(1+norm(c_w))>= eps_r_cw \
        or norm(r_cz)/(1+norm(c_z)) >= eps_r_cz or mu>= eps_mu):
            print('random_progression \n \n \n')
            big_A = hstack((A_w,A_z))
            D_sqrt = diags(np.sqrt(np.append(1/Q_w.diagonal().ravel(),1/(Q_z+Theta_inv).diagonal().ravel()))).tocsr()
            Hessian_sqrt = big_A.dot(D_sqrt)
            #S = self.gen_sketch_matrix(big_A.shape[1],1000)#big_A.shape[0])
            S = np.random.randn(big_A.shape[1],5000)
            SH = Hessian_sqrt.dot(S)
            sketched_hessian = SH.dot(SH.T)
            #from numpy import linalg as LA
            #w, v = LA.eig(sketched_hessian)
            #print(w)
            L = cholesky(sketched_hessian)
            d_y_aff_random = dense_inv(L).dot(dense_inv(L.T).dot(r_b_hat.T.ravel()))

            #####compare
            Hessian = Hessian_sqrt.dot(Hessian_sqrt.T)
            L_exact = cholesky(Hessian.toarray())
            d_y_aff_exact = dense_inv(L_exact).dot(dense_inv(L_exact.T).dot(r_b_hat.T.ravel()))
            d_w_aff_exact = diags(1/Q_w.diagonal().ravel()).tocsr().dot(A_w.T.dot(d_y_aff_exact) - r_cw)
            d_z_aff_exact = diags(1/(Q_z+Theta_inv).diagonal().ravel()).tocsr().dot(A_z.T.dot(d_y_aff_exact) - r_cz_hat)
            print('difference of solutions to linear equation ' )
            print(norm(d_y_aff_exact-d_y_aff_random)/d_y_aff_random.shape[0],'\n')


            #########
            d_w_aff_random = diags(1/Q_w.diagonal().ravel()).tocsr().dot(A_w.T.dot(d_y_aff_random) - r_cw)
            d_z_aff_random = diags(1/(Q_z+Theta_inv).diagonal().ravel()).tocsr().dot(A_z.T.dot(d_y_aff_random) - r_cz_hat)
            d_s_aff_random = -(s + s / z * d_z_aff_random)
            d_v_aff_random = -v + v / (u - z) * d_z_aff_random
            print('test affine_random')
            print('r_b_exact', norm(A_w.dot(w+d_w_aff_exact) + A_z.dot(z+d_z_aff_exact)- b))
            print('r_b_random', norm(A_w.dot(w+d_w_aff_random) + A_z.dot(z+d_z_aff_random)- b))
            print('r_cw_random',norm( -Q_w.dot(d_w_aff_random) +A_w.T.dot(d_y_aff_random)-r_cw))
            print('r_cz_random',norm( -(Q_z+Theta_inv).dot(d_z_aff_random)+A_z.T.dot(d_y_aff_random)-r_cz_hat))
            print('end')
            print('\n')
            alpha_aff_p_random = 1
            alpha_aff_d_random = 1
            idx_z_random = np.where(d_z_aff_random < 0)[0]
            idx_uz_random = np.where(d_z_aff_random > 0)[0]
            idx_s_random = np.where(d_s_aff_random < 0)[0]
            idx_v_random = np.where(d_v_aff_random < 0)[0]
            if idx_z_random.size != 0:
                alpha_aff_p_random = min(alpha_aff_p_random, np.min(-z[idx_z_random] / d_z_aff_random[idx_z_random]))
            if idx_uz_random.size != 0:
                alpha_aff_p_random = min(alpha_aff_p_random, np.min((u-z)[idx_uz_random] /d_z_aff_random[idx_uz_random]))
            if idx_s_random.size != 0:
                alpha_aff_d_random = min(alpha_aff_d_random, np.min(-s[idx_s_random] / d_s_aff_random[idx_s_random]))
            if idx_v_random.size != 0:
                alpha_aff_d_random = min(alpha_aff_d_random, np.min(-v[idx_v_random] / d_v_aff_random[idx_v_random]))
            print('affine step length')
            print('alpha_aff_p', alpha_aff_p_random)
            print('alpha_aff_d', alpha_aff_d_random)
            print('\n')
            z_aff_random = z + d_z_aff_random * alpha_aff_p_random
            s_aff_random = s + d_s_aff_random * alpha_aff_d_random
            v_aff_random = v + d_v_aff_random * alpha_aff_d_random
            y_aff_random = y + d_y_aff_random * alpha_aff_d_random
            w_aff_random = w + d_w_aff_random * alpha_aff_p_random
            assert (z_aff_random>=-1e-8).all() ,'%d' %z_aff_random[np.where(z_aff_random<=0)]
            assert (s_aff_random>=-1e-8).all(),'%d' %s_aff_random[np.where(s_aff_random<=0)]
            assert (v_aff_random>=-1e8).all(), '%d' %v_aff_random[np.where(v_aff_random<=0)]
            assert (u-z_aff_random>=-1e-6).all(), '%d' %(u-z_aff_random)[np.where((u-z_aff_random)<=0)]
            mu_aff_random = ((s_aff_random).dot(z_aff_random) + (u - z_aff_random).dot(v_aff_random)) / (2 * self.n)

            # Determine centering parameter
            
            sigma_random = (mu_aff_random / mu) ** 3
            mu_t_random = sigma_random * mu
            e = np.ones(z.shape[0])
            r_sz_random = -z*s-d_z_aff_random*d_s_aff_random+mu_t_random*e
            r_vz_random = -v*(u-z)+d_z_aff_random*d_v_aff_random+mu_t_random*np.ones(z.shape[0])
            r_cz_random = -A_z.T.dot(y) +c_z +Q_z.dot(z)-s+v
            r_cw_random = r_cw
            r_cz_hat_random = r_cz_random-r_sz_random/z+r_vz_random/(u-z)#Theta_inv.dot(z))
            r_b_hat_random = r_b+A_w.dot(inv(Q_w)).dot(r_cw)+A_z.dot(diags(1/(Q_z+Theta_inv).diagonal().ravel()).tocsr()).dot(r_cz_hat)


            d_y_pre_corr_random = dense_inv(L).dot(dense_inv(L.T).dot(r_b_hat_random.ravel()))
            d_w_pre_corr_random = inv(Q_w).dot(A_w.T.dot(d_y_pre_corr_random) - r_cw_random)
            d_z_pre_corr_random = diags(1/(Q_z+Theta_inv).diagonal().ravel()).tocsr().dot(A_z.T.dot(d_y_pre_corr_random) - r_cz_hat_random)
            d_s_pre_corr_random = r_sz_random/z- s*d_z_pre_corr_random/z
            d_v_pre_corr_random = r_vz_random/(u-z) +v*d_z_pre_corr_random/(u-z)
            print('test correct')
            print('r_b', norm(A_w.dot(w+d_w_pre_corr_random) + A_z.dot(z+d_z_pre_corr_random)- b))
            print('r_cw',norm( -Q_w.dot(d_w_pre_corr_random) +A_w.T.dot(d_y_pre_corr_random)-r_cw_random))
            print('r_cz',norm( -(Q_z+Theta_inv).dot(d_z_pre_corr_random)+A_z.T.dot(d_y_pre_corr_random)-r_cz_hat_random))
            print('end')
            print('\n')
            # Compute predictor step alpha_p
            alpha_p_random = 1
            alpha_d_random = 1
            idx_z_random = np.where(d_z_pre_corr_random < 0)[0]
            idx_uz_random = np.where(d_z_pre_corr_random > 0)[0]
            idx_s_random = np.where(d_s_pre_corr_random < 0)[0]
            idx_v_random = np.where(d_v_pre_corr_random < 0)[0]
            if idx_z_random.size != 0:
                alpha_p_random = min(alpha_p_random, np.min(-z[idx_z_random] / d_z_pre_corr_random[idx_z_random]))
            if idx_uz_random.size != 0:
                alpha_p_random = min(alpha_p_random, np.min((u-z)[idx_uz_random] /d_z_pre_corr_random[idx_uz_random]))
            if idx_s_random.size != 0:
                alpha_d_random = min(alpha_d_random, np.min(-s[idx_s_random] / d_s_pre_corr_random[idx_s_random]))
            if idx_v_random.size != 0:
                alpha_d_random = min(alpha_d_random, np.min(-v[idx_v_random] / d_v_pre_corr_random[idx_v_random]))

            print()
            print('correct step length')
            print('alpha_p', alpha_p_random)
            print('alpha_d', alpha_d_random)
            print('\n')
            d_w_random = d_w_pre_corr_random
            d_z_random = d_z_pre_corr_random
            d_s_random = d_s_pre_corr_random
            d_v_random = d_v_pre_corr_random
            d_y_random = d_y_pre_corr_random

            w += d_w_random * eta * alpha_p_random
        # _hat
            z += d_z_random * eta * alpha_p_random
        # _hat
            s += d_s_random * eta * alpha_d_random
            # _hat
            v += d_v_random * eta * alpha_d_random
                # _hat
            y += d_y_random * eta * alpha_d_random

            print('###############')
            print('result')
            r_b = -(A_w.dot(w) + A_z.dot(z) - b)
            print('r_b',norm(r_b)/(1+norm(self.b)))
            #print('d_w',d_w)
            #print('r_b',r_b)
            r_cw = -(-Q_w.dot(w) + A_w.T.dot(y) - c_w)
            print('r_cw',norm(r_cw)/(1+norm(c_w)))
            r_sz = -z*s
            r_vz = -(u-z)*v
            r_cz = -A_z.T.dot(y) +c_z +Q_z.dot(z)-s+v
            r_cz_hat = r_cz-r_sz/z+r_vz/(u-z)#Theta_inv.dot(z))
            print('r_cz',norm(r_cz)/(1+norm(r_cz)))
            #print('r_cz',r_cz)
            print('mu',mu)
            print('\n')
        # 2n((z0)Ts0+(u−z0)Tv0)
            Theta_inv = csc_matrix(s / z + v / (u - z))
            Theta_inv = diags(Theta_inv.A.ravel()).tocsc()#.toarray()
            #print('Theta_inv',Theta_inv)
            mu = (s.dot(z) + (u - z).dot(v)) / (2 * self.n)
            #print('r_b_hat',r_b_hat)
            r_b_hat = r_b+A_w.dot(inv(Q_w)).dot(r_cw)+A_z.dot(diags(1/(Q_z+Theta_inv).diagonal().ravel()).tocsr()).dot(r_cz_hat)
            #print('r_b_hat_hat',r_b_hat_hat)
            k+= 1



        return w,z,error_log,k



            #####random part##############
if __name__ == '__main__':
    c = SVM(tao=100)
    c.load_data('a7a.txt')
    Q_w,Q_z,A_w,A_z,c_w,c_z,b,u = c.data_preprocess()
    w = np.ones(c.m)
    z = 2*np.ones(c.n)
    s = np.ones(c.n)
    v = np.ones(c.n)
    y = np.ones(c.m+1)
    w,z,log,k =c.fit(Q_w,Q_z,A_w,A_z,c_w,c_z,b,u,init= (w,z,y,s,v))
            
