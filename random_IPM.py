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
        #print('Theta_inv computed')
        r_b = -(A_w.dot(w) + A_z.dot(z) - b)
        #print('r_b computed')
        r_cw = -(-Q_w.dot(w) + A_w.T.dot(y) - c_w)
        #print('r_cw computed')
        r_sz = -z*s
        r_vz = -(u-z)*v
        #print('r_sz,r_vz computed')
        r_cz = -A_z.T.dot(y) +c_z +Q_z.dot(z)-s+v
       # print('r_cz computed')
        r_cz_hat = r_cz-r_sz/z+r_vz/(u-z)
       # print('r_cz_hat computed')
        mu = (s.dot(z) + (u - z).dot(v)) / (2 * self.n)
       # print('mu computed')
#r_b_hat = r_b + A_z.dot(Q_z + Theta_inv).dot(r_cz)
#print('r_b_hat',r_b_hat)
# print(A_w.dot(inv(Q_w)).dot(r_cw).shape)
        temp_A_z = A_z.dot(diags(1/(Q_z+Theta_inv).diagonal().ravel()).tocsr())
        r_b_hat = r_b+A_w.dot(inv(Q_w)).dot(r_cw)+temp_A_z.dot(r_cz_hat)          #A_z.dot(inv(Q_z+Theta_inv)).dot(r_cz_hat)
       # print('r_b_hat computed')
        error_log = list()
        error_exact = list()
        error_random = list()
        print('residuals computed')
        while k<= maxk and( norm(r_b)/(1+norm(self.b))>= eps_r_b or norm(r_cw)/(1+norm(c_w))>= eps_r_cw \
        or norm(r_cz)/(1+norm(c_z)) >= eps_r_cz or mu>= eps_mu):
            error_log.append(mu)            
            print(k)
            M_w = A_w.dot(inv(Q_w)).dot(A_w.T)
            M_z = A_z.dot(diags(1/(Q_z+Theta_inv).diagonal().ravel()).tocsr()).dot(A_z.T)
            M = (M_w + M_z).toarray()
            L = cholesky(M)
            d_y_aff = dense_inv(L).dot(dense_inv(L.T).dot(r_b_hat.T.ravel()))
            d_w_aff = diags(1/Q_w.diagonal().ravel()).tocsr().dot(A_w.T.dot(d_y_aff) - r_cw)
            d_z_aff = diags(1/(Q_z+Theta_inv).diagonal().ravel()).tocsr().dot(A_z.T.dot(d_y_aff) - r_cz_hat)
            d_s_aff = -(s + s / z * d_z_aff)
            d_v_aff = -v + v / (u - z) * d_z_aff
            '''
            print('test affine')
            print('r_b', norm(A_w.dot(w+d_w_aff) + A_z.dot(z+d_z_aff)- b))
            print('r_cw',norm( -Q_w.dot(d_w_aff) +A_w.T.dot(d_y_aff)-r_cw))
            print('r_cz',norm( -(Q_z+Theta_inv).dot(d_z_aff)+A_z.T.dot(d_y_aff)-r_cz_hat))
            print('end')
            print('\n')
            '''
            alpha_aff_p = 1
            alpha_aff_d = 1
            idx_z = np.where(d_z_aff < 0)[0]
            idx_uz = np.where(d_z_aff > 0)[0]
            idx_s = np.where(d_s_aff < 0)[0]
            idx_v = np.where(d_v_aff < 0)[0]
            if idx_z.size != 0:
                alpha_aff_p = min(alpha_aff_p, np.min(-z[idx_z] / d_z_aff[idx_z]))
            if idx_uz.size != 0:
                alpha_aff_p = min(alpha_aff_p, np.min((u-z)[idx_uz] /d_z_aff[idx_uz]))
            if idx_s.size != 0:
                alpha_aff_d = min(alpha_aff_d, np.min(-s[idx_s] / d_s_aff[idx_s]))
            if idx_v.size != 0:
                alpha_aff_d = min(alpha_aff_d, np.min(-v[idx_v] / d_v_aff[idx_v]))
            print('affine step length')
            print('alpha_aff_p', alpha_aff_p)
            print('alpha_aff_d', alpha_aff_d)
            print('\n')
            z_aff = z + d_z_aff * alpha_aff_p
            s_aff = s + d_s_aff * alpha_aff_d
            v_aff = v + d_v_aff * alpha_aff_d
            y_aff = y + d_y_aff * alpha_aff_d
            w_aff = w + d_w_aff * alpha_aff_p
            assert (z_aff>=-1e-8).all() ,'%d' %z_aff[np.where(z_aff<=0)]
            assert (s_aff>=-1e-8).all(),'%d' %s_aff[np.where(s_aff<=0)]
            assert (v_aff>=-1e8).all(), '%d' %v_aff[np.where(v_aff<=0)]
            assert (u-z_aff>=-1e-6).all(), '%d' %(u-z_aff)[np.where((u-z_aff)<=0)]
            mu_aff = ((s_aff).dot(z_aff) + (u - z_aff).dot(v_aff)) / (2 * self.n)            
            # Determine centering parameter
            sigma = (mu_aff / mu) ** 3
            mu_t = sigma * mu
            e = np.ones(z.shape[0])
            r_sz = -z*s-d_z_aff*d_s_aff+mu_t*e
            r_vz = -v*(u-z)+d_z_aff*d_v_aff+mu_t*np.ones(z.shape[0])
            r_cz = -A_z.T.dot(y) +c_z +Q_z.dot(z)-s+v
            r_cz_hat = r_cz-r_sz/z+r_vz/(u-z)#Theta_inv.dot(z))
            r_b_hat = r_b+A_w.dot(inv(Q_w)).dot(r_cw)+A_z.dot(diags(1/(Q_z+Theta_inv).diagonal().ravel()).tocsr()).dot(r_cz_hat)
            #print(r_b_hat)

            #inv(Q_z+Theta_inv)).dot(r_cz_hat)
        # Solve system
            #d_y_pre_corr = dense_inv(M).dot(r_b_hat.T.ravel())
            d_y_pre_corr = dense_inv(L).dot(dense_inv(L.T).dot(r_b_hat.ravel()))
            d_w_pre_corr = inv(Q_w).dot(A_w.T.dot(d_y_pre_corr) - r_cw)
            d_z_pre_corr = diags(1/(Q_z+Theta_inv).diagonal().ravel()).tocsr().dot(A_z.T.dot(d_y_pre_corr) - r_cz_hat)
            d_s_pre_corr = r_sz/z- s*d_z_pre_corr/z
            d_v_pre_corr = r_vz/(u-z) +v*d_z_pre_corr/(u-z)
            print('test correct')
            print('r_b', norm(A_w.dot(w+d_w_pre_corr) + A_z.dot(z+d_z_pre_corr)- b))
            print('r_cw',norm( -Q_w.dot(d_w_pre_corr) +A_w.T.dot(d_y_pre_corr)-r_cw))
            print('r_cz',norm( -(Q_z+Theta_inv).dot(d_z_pre_corr)+A_z.T.dot(d_y_pre_corr)-r_cz_hat))
            print('end')
            print('\n')
            # Compute predictor step alpha_p
            alpha_p = 1
            alpha_d = 1
            idx_z = np.where(d_z_pre_corr < 0)[0]
            idx_uz = np.where(d_z_pre_corr > 0)[0]
            idx_s = np.where(d_s_pre_corr < 0)[0]
            idx_v = np.where(d_v_pre_corr < 0)[0]
            if idx_z.size != 0:
                alpha_p = min(alpha_p, np.min(-z[idx_z] / d_z_pre_corr[idx_z]))
            if idx_uz.size != 0:
                alpha_p = min(alpha_p, np.min((u-z)[idx_uz] /d_z_pre_corr[idx_uz]))
            if idx_s.size != 0:
                alpha_d = min(alpha_d, np.min(-s[idx_s] / d_s_pre_corr[idx_s]))
            if idx_v.size != 0:
                alpha_d = min(alpha_d, np.min(-v[idx_v] / d_v_pre_corr[idx_v]))
            print()
            print('correct step length')
            print('alpha_p', alpha_p)
            print('alpha_d', alpha_d)
            print('\n')
            d_w = d_w_pre_corr
            d_z = d_z_pre_corr
            d_s = d_s_pre_corr
            d_v = d_v_pre_corr
            d_y = d_y_pre_corr

            w += d_w * eta * alpha_p
        # _hat
            z += d_z * eta * alpha_p
        # _hat
            s += d_s * eta * alpha_d
            # _hat
            v += d_v * eta * alpha_d
                # _hat
            y += d_y * eta * alpha_d
           # print('z',z)
       # print('s',s)
           # print('w',w)
        #print('v',v)
       # print('y',y)
    
#print(A_z.shape)
#print('r_b',r_b)
        
#print('r_cw',r_cw)
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

if __name__ == '__main__':
    c = SVM(tao=100)
    c.load_data('a9a.txt')
    Q_w,Q_z,A_w,A_z,c_w,c_z,b,u = c.data_preprocess()
    w = np.ones(c.m)
    z = 2*np.ones(c.n)
    s = np.ones(c.n)
    v = np.ones(c.n)
    y = np.ones(c.m+1)
    w,z,log,k =c.fit(Q_w,Q_z,A_w,A_z,c_w,c_z,b,u,init= (w,z,y,s,v))

