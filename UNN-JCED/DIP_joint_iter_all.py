import numpy as np
import torch
from torch import nn 
          
def arr(*args):
    # 1 input for python array to numpy array
    if len(args) == 1:
        return np.array(args[0]);
    # 2 inputs: start, end, step = 1
    if len(args) == 2:
        return np.arange(args[0], args[1] + 1, 1); 
    # 3 inputs: start, end, step
    if len(args) == 3:
        return np.arange(args[0], args[1] + 1, args[2]); 
def matrix_diag_t(diags_ndim):
    batch_size = diags_ndim.shape[0];
    matlen =  diags_ndim.shape[1];
    
    res = np.zeros((batch_size, matlen, matlen), dtype=np.complex128);
    
    for i in range(0, batch_size):
        res[i, :] = np.diag(diags_ndim[i, :]);
    return res
def fft_t(mat):
    return np.fft.fft(mat, axis=1);
def ifft_t(mat):
    return np.fft.ifft(mat, axis=1);
def arrange3d(batch_size, start, end):
    tmp = np.arange(start, end+1, 1);
    return np.tile(tmp, (batch_size, 1) ) 
def circshift2d_colwise(array, steps):
    array = torch.roll(array, int(steps), dims = 0);
    return array
def circshift3d_colwise(array, steps):
    for i in range(0, array.shape[0]):
        array[i, :] = torch.roll(array[i, :], int(steps[i]), dims = 0);
    return array;
def kron3D(a,b,batch_size):
    sh=a.shape[-1]*b.shape[-1]
    out = torch.einsum('aij,akl->aikjl',a,b).reshape(batch_size,sh,sh)
    return out
def DFTMat(N,batch_size):
    Wn = np.exp(-1j*2*np.pi/N);
    colVec = np.expand_dims(arr(0, N-1), 0).T;
    rowVec = np.expand_dims(arr(0, N-1), 0);
    dftmat = np.power(Wn, np.matmul(colVec, rowVec));
    dftmat_norm = np.tile(np.expand_dims(1/np.sqrt(N)*dftmat,0),(batch_size,1,1));
    return dftmat_norm;
def IDFTMat(N,batch_size):
    Wn = np.exp(-1j*2*np.pi/N);
    colVec = np.expand_dims(arr(0, N-1), 0).T;
    rowVec = np.expand_dims(arr(0, N-1), 0);
    idftmat = np.power(Wn, (-1)*np.matmul(colVec, rowVec));
    idftmat_norm = 1/np.sqrt(N)*idftmat;
    idftmat_norm = np.tile(np.expand_dims(idftmat_norm,0),(batch_size,1,1));
    return idftmat_norm;

def Regen_H(N,M,doppler,delay,hi,dtype,device):
 
    Heff_rect = (torch.zeros((M*N, M*N))+1j*torch.zeros(( M*N, M*N))) .type(dtype).to(device)
    dftmat = torch.from_numpy(DFTMat(N, 1)).type(dtype)
    idftmat = torch.from_numpy(IDFTMat(N, 1)).type(dtype)
    num_paths = doppler.shape[0] 
    for itap in range(0,num_paths): 
        li_val = torch.unsqueeze(delay[itap], axis=-1); 
        ki_val = torch.unsqueeze(doppler[itap], axis=-1) 
        hi_val = torch.unsqueeze(hi[itap], axis=-1);
        if abs(hi_val)>0:
            # delay mat 
            piMati_t = circshift3d_colwise(torch.tile(torch.eye(M*N),(1,1,1)), li_val).type(dtype) 
            
            # doppler mat
            deltaMat_diag = np.exp(1j*2*torch.pi*torch.unsqueeze(ki_val, axis=1)/(M*N)*arrange3d(1, 0, M*N-1));
            deltaMati = torch.diagflat(deltaMat_diag).type(dtype) 
            # generate Pi
            p1 = kron3D(dftmat, torch.tile(torch.eye(M),(1,1,1)),1).type(dtype) 
            Pi = torch.matmul(p1, piMati_t).type(dtype) 
            # generate Qi
            Qi = torch.matmul(deltaMati, kron3D(idftmat, torch.tile(torch.eye(M).type(dtype) ,(1,1,1)),1)).type(dtype) 
            # generate Ti
            Ti = torch.matmul(Pi, Qi).type(dtype).to(device) 
            Heff_rect = Heff_rect + (hi_val*Ti).squeeze();
             
    return Heff_rect 

 
def Bulid_Heff(h_est,lmax,kmax,N,M,dtype,device):  
  
    h_est_r = h_est[0:(2*kmax+1)*(lmax+1)]
    h_est_i = h_est[(2*kmax+1)*(lmax+1):]
    
    h_complex_torch = h_est_r+1j*h_est_i
 
    lis = np.kron(np.arange(lmax + 1), np.ones(2 * kmax + 1, dtype=int))  # the delays on all possible paths
    kis = np.tile(np.arange(-kmax, kmax + 1), lmax + 1)  # the dopplers on all possible paths 
    Del_ind = torch.from_numpy(lis)
    Dop_ind = torch.from_numpy(kis)
    H_full = Regen_H(N,M,Dop_ind,Del_ind,h_complex_torch,dtype,device)   
 

    H_full_r = torch.real(H_full);
    H_full_i = torch.imag(H_full);
    H_full_real = torch.concatenate([torch.concatenate([H_full_r, -H_full_i], axis=1), torch.concatenate([H_full_i, H_full_r], axis=1)], axis=0) 
    
         
    return H_full,H_full_real
 
class DD_ce(nn.Module):
    def __init__(self,lmax,kmax):
        super().__init__()          
        self.lenh = (lmax+1)*(2*kmax+1)  
        self.nn1 = nn.Linear(4, 8)
        self.nn2 = nn.Linear(8, 16)   
        self.nn3 = nn.Linear(16, 32) 
        self.nn4 = nn.Linear(32, 2*self.lenh)   
        self.act1 = nn.ReLU()     
        nn.init.xavier_uniform_(self.nn1.weight)
        nn.init.xavier_uniform_(self.nn2.weight)
        nn.init.xavier_uniform_(self.nn3.weight)
        nn.init.xavier_uniform_(self.nn4.weight)
        
        nn.init.uniform_(self.nn1.bias,-1/np.sqrt(self.nn1.bias.shape[0]), 1/np.sqrt(self.nn1.bias.shape[0])) 
        nn.init.uniform_(self.nn2.bias,-1/np.sqrt(self.nn2.bias.shape[0]), 1/np.sqrt(self.nn2.bias.shape[0]))
        nn.init.uniform_(self.nn3.bias,-1/np.sqrt(self.nn3.bias.shape[0]), 1/np.sqrt(self.nn3.bias.shape[0]))
        nn.init.uniform_(self.nn4.bias,-1/np.sqrt(self.nn4.bias.shape[0]), 1/np.sqrt(self.nn4.bias.shape[0]))
        
        
    def forward(self,x):    
        o1 = self.act1(self.nn1(x)) 
        o2 = self.act1(self.nn2(o1)) 
        o3 = self.act1(self.nn3(o2)) 
        o4 = self.nn4(o3) 
        return o4  
 
class DD_det(nn.Module):
    def __init__(self,user_num):
        super().__init__() 
        self.nn1 = nn.Linear(4, 8)
        self.nn2 = nn.Linear(8, 16)   
        self.nn3 = nn.Linear(16, 32) 
        self.nn4 = nn.Linear(32,user_num) 
         
        nn.init.xavier_uniform_(self.nn1.weight)
        nn.init.xavier_uniform_(self.nn2.weight)
        nn.init.xavier_uniform_(self.nn3.weight)
        nn.init.xavier_uniform_(self.nn4.weight)
        nn.init.uniform_(self.nn1.bias,-1/np.sqrt(self.nn1.bias.shape[0]), 1/np.sqrt(self.nn1.bias.shape[0])) 
        nn.init.uniform_(self.nn2.bias,-1/np.sqrt(self.nn2.bias.shape[0]), 1/np.sqrt(self.nn2.bias.shape[0]))
        nn.init.uniform_(self.nn3.bias,-1/np.sqrt(self.nn3.bias.shape[0]), 1/np.sqrt(self.nn3.bias.shape[0]))
        nn.init.uniform_(self.nn4.bias,-1/np.sqrt(self.nn4.bias.shape[0]), 1/np.sqrt(self.nn4.bias.shape[0]))
          
        self.act1 = nn.ReLU()  
        self.act2 = nn.Tanh()   
    def forward(self,x):  
        
        o1 = self.act1(self.nn1(x)) 
        o2 = self.act1(self.nn2(o1)) 
        o3 = self.act1(self.nn3(o2)) 
        o4 = self.act2(self.nn4(o3)) 
        return o4
   
class DIP_joint(object):
    
    def __init__(self,M,LR,batch_size):
         
        self.M = M 
        self.LR = LR
        self.batch_size = batch_size 
        constellation = np.linspace(int(-np.sqrt(M) + 1), int(np.sqrt(M) - 1), int(np.sqrt(M)))
        alpha = np.sqrt((constellation ** 2).mean())
        constellation /= (alpha * np.sqrt(2))
        self.constellation = constellation 
        
    def calculate_mean_var(self,pyx):
        constellation_expanded = np.expand_dims(self.constellation, axis=1)
        constellation_expanded_repeat = np.repeat(constellation_expanded.transpose(1,0), pyx.shape[0], axis=0)
        mean = np.matmul(pyx, constellation_expanded)
        var = np.square(np.abs(constellation_expanded_repeat - mean))
        var = np.multiply(pyx, var) 
        var = np.sum(var, axis=1)
        
        return mean, var
    
    def calculate_pyx( self,mean, var):
        var = var.reshape(-1,1)
        constellation_expanded = np.expand_dims(self.constellation, axis=0)
        constellation_expanded_repeat = np.repeat(constellation_expanded, mean.shape[0], axis=0)
        arg_1 = np.square(np.abs(constellation_expanded_repeat - mean))
        log_pyx = (-1 * arg_1)/(2*var)
        log_pyx = log_pyx - np.expand_dims(np.max(log_pyx,1),1)
        p_y_x = np.exp(log_pyx)
        p_y_x = p_y_x/(np.expand_dims(np.sum(p_y_x, axis=1),1))
        
        return p_y_x    
    def BPIC(self,x_init,H_full,y_full,num_iter,sigma2,x_real):
        
        H = H_full
        y = y_full
        diff_list = []
        user_num = x_init.shape[0]
        sigma2 = np.repeat(sigma2.reshape(-1,1), user_num,axis=-1)
        I_mat  = np.tile(np.eye(user_num),[1,1])
        W = (I_mat+1) - I_mat*2
        H_2 = np.matmul(H.transpose(1,0),H)
        invD_H_vec = 1/ (np.diagonal(H_2,axis1=0, axis2=1))
        invD_H_diag = np.nan_to_num((1/H_2)) *I_mat 
        HTy = np.matmul(H.transpose(1,0),np.expand_dims(y,1))
        x_app = np.expand_dims(x_init, axis=-1) 
        Var_BSE_DSC = np.zeros((user_num,1)) 
        var_PIC_prev = None
        x_PIC_prev = None
        x_hat_prev =None
        V_BSE_prev =None
        Var_DSC_prev = None
        ser_iter = np.zeros(num_iter)
        diff = None
        for iteration in range(num_iter):
            # BSO
            approx_error_estim = np.matmul(np.square((W*H_2)),Var_BSE_DSC ) 
            var_PIC = sigma2 * invD_H_vec  + np.squeeze(approx_error_estim) / ( np.square(np.diagonal(H_2,axis1=0, axis2=1))) 
            x_PIC = np.matmul( invD_H_diag, (HTy - np.matmul((W*H_2),x_app)))
            # Avoiding negative lamda and gamma
            if np.any(var_PIC < 0):
                indices = np.where(var_PIC<0)
                var_PIC[indices]=var_PIC_prev[indices]
                x_PIC[indices]=x_PIC_prev[indices]
            var_PIC_prev = var_PIC
            x_PIC_prev = x_PIC
            # Calculating P_y_x
            p_y_x = self.calculate_pyx (x_PIC, var_PIC)
            # Calculating mean and variance of \hat{P}_x_y
            x_hat, V_BSE = self.calculate_mean_var(p_y_x)
            V_BSE = np.clip(np.expand_dims(V_BSE,1), 1e-13, None) 
            err_DSC = HTy - np.matmul(H_2,x_hat)
            Var_DSC = np.square(np.matmul(invD_H_diag,err_DSC) )
            if iteration == 0:                
                x_app = x_hat
                Var_BSE_DSC = V_BSE
            else:
                # Computing dsc
                V_DSC_tot = Var_DSC + Var_DSC_prev
                rho =Var_DSC/V_DSC_tot
                x_app = rho* x_hat_prev + (1-rho) * x_hat
                Var_BSE_DSC = rho* V_BSE_prev + (1-rho) * V_BSE
            if iteration>0:
                diff = ((x_app -x_hat_prev)**2).mean()
                diff_list.append(diff)
            #Storing Values    
            x_hat_prev= x_app
            V_BSE_prev = Var_BSE_DSC
            Var_DSC_prev = Var_DSC
              
            ser_iter[iteration] = self.ser(x_app.squeeze(), x_real.squeeze()) 
        return x_app,ser_iter,iteration
 
    def QAM_const(self):
        mod_n = self.M
        sqrt_mod_n = int(np.sqrt(mod_n))
        real_qam_consts = np.empty((mod_n), dtype=np.int64)
        imag_qam_consts = np.empty((mod_n), dtype=np.int64)
        for i in range(sqrt_mod_n):
            for j in range(sqrt_mod_n):
                    index = sqrt_mod_n*i + j
                    real_qam_consts[index] = i
                    imag_qam_consts[index] = j
                    
        return(self.constellation[real_qam_consts], self.constellation[imag_qam_consts])
    
    def ser(self,x_hat, x_true): 
        real_QAM_const,imag_QAM_const = self.QAM_const()
        x_real, x_imag = np.split(x_hat, 2, -1)
        x_real = np.expand_dims(x_real,-1).repeat(real_QAM_const.size,-1)
        x_imag = np.expand_dims(x_imag,-1).repeat(imag_QAM_const.size,-1)
    
        x_real = np.power(x_real - real_QAM_const, 2)
        x_imag = np.power(x_imag - imag_QAM_const, 2)
        x_dist = x_real + x_imag
        estim_indices = np.argmin(x_dist, axis=-1)
        
        x_real_true, x_imag_true = np.split(x_true, 2, -1)
        x_real_true = np.expand_dims(x_real_true,-1).repeat(real_QAM_const.size,-1)
        x_imag_true = np.expand_dims(x_imag_true,-1).repeat(imag_QAM_const.size,-1)
    
        x_real_true = np.power(x_real_true - real_QAM_const, 2)
        x_imag_true = np.power(x_imag_true - imag_QAM_const, 2)
        x_dist_true = x_real_true + x_imag_true
        true_indices = np.argmin(x_dist_true, axis=-1)
        ser = np.sum(true_indices!=estim_indices)/true_indices.size
        return ser
    
    def Build_phi_new(self, lmax, kmax, M, N, x_est, device):
        num_data_sym_complex = x_est.shape[0] // 2
        x_complex = x_est[:num_data_sym_complex] + 1j * x_est[num_data_sym_complex:]
        X = x_complex.reshape(M, N)
         
        lis = np.kron(np.arange(lmax + 1), np.ones(2 * kmax + 1, dtype=int))
        kis = np.tile(np.arange(-kmax, kmax + 1), lmax + 1)
      
        lis_torch = torch.from_numpy(lis).to(device)
        kis_torch = torch.from_numpy(kis).to(device)
         
        # Generate all yk and yl combinations
        yk, yl = torch.meshgrid(
            torch.arange(M, device=device), 
            torch.arange(N, device=device), 
            indexing='ij'
        )
        yk = yk.reshape(-1, 1)  # Shape (M*N, 1)
        yl = yl.reshape(-1, 1)  # Shape (M*N, 1)
        
        # Expand li and ki to (1, pmax)
        li = lis_torch.view(1, -1)  # Shape (1, pmax)
        ki = kis_torch.view(1, -1)  # Shape (1, pmax)
        
        # Compute xl with modulo handling
        xl = yl - li
        mask_yl_li = (yl < li)
        xl += mask_yl_li.long() * N
        
        # Compute xk with modulo N
        xk = (yk - ki) % N
        
        # Calculate pss_beta components
        yl_li_float = (yl.float() - li.float())
        ki_float = ki.float()
        exponent_base = 2j * torch.pi * yl_li_float * ki_float / (N * M)
        base = torch.exp(exponent_base)
        
        xk_float = xk.float()
        exponent_extra = -2j * torch.pi * xk_float / M
        extra = torch.exp(exponent_extra)
        
        # Apply condition using mask
        pss_beta = base * torch.where(mask_yl_li, extra, torch.tensor(1.0, device=device, dtype=torch.complex64))
        
        # Indexing X matrix
        xk = xk.long()
        xl = xl.long()
        X_elements = X[xk, xl]
        
        # Compute Phi_torch in one step
        Phi_torch = X_elements * pss_beta
        
        # Construct real-valued Phi matrix
        Phi_r = torch.real(Phi_torch)
        Phi_i = torch.imag(Phi_torch)
        Phi_real = torch.cat([
            torch.cat([Phi_r, -Phi_i], dim=1),
            torch.cat([Phi_i, Phi_r], dim=1)
        ], dim=0) 
        return Phi_real
 
    def DIP_joint_alter_new(self,y_full_real,phi_p_full_real,H_thre_full_real,h_thre_real,h_vec_real,xd_real,xp_real,sigma2_d,sigma2_p,lmax,kmax,N,M,iter_outer,iter1,iter2,iter_bpic,noise_app):
     
        dtype =  torch.float32
        device = torch.device('cuda', index=0)    
        
        NMSE_hest = np.zeros((self.batch_size,iter_outer*iter1)) 
        SER_dip = np.zeros((self.batch_size,iter_outer*iter2))
        SER_bpic = np.zeros((self.batch_size,iter_outer*iter_bpic))
        dip_h_arr = np.zeros((self.batch_size,iter_outer*iter1,h_vec_real.shape[1])) 
        dip_x_arr = np.zeros((self.batch_size,iter_outer*iter2,xd_real.shape[1]))
        x_bpic_array = np.zeros((self.batch_size,iter_outer,xd_real.shape[1])) 
        noise_app_arr =  np.zeros((self.batch_size,iter_outer))
        thre_list = np.linspace(0, sigma2_p,iter_outer*iter1) 
        iter_NN1 = iter1     
        iter_NN2 = iter2  
        for bs in range(self.batch_size):  
             
            # net1  = DD_ce_thre(lmax,kmax,).type(dtype).to(device)   
            net1  = DD_ce(lmax,kmax).type(dtype).to(device)   
            net2  = DD_det(xd_real.shape[-1]).type(dtype).to(device)        
            y_full_torch = torch.from_numpy(y_full_real[bs].squeeze()).type(dtype).to(device).unsqueeze(-1)  
            phi_p_full_torch = torch.from_numpy(phi_p_full_real[bs].squeeze()).type(dtype).to(device)    
            xp_torch = torch.from_numpy(xp_real[bs].squeeze()).type(dtype).to(device)     
            mse = torch.nn.MSELoss().type(dtype).to(device)   
            input_size1 =  4
            input_size2 = 4 
            z1 = torch.randn(input_size1).type(dtype).to(device)  
            z2 = torch.randn(input_size2).type(dtype).to(device)   
            optimizer1 = torch.optim.Adam(net1.parameters(),lr=self.LR) 
            optimizer2 = torch.optim.Adam(net2.parameters(),lr=self.LR)   
            h_est_np = None
            phi_x_np_torch = 0     
            sigma2_app = np.float64(1.0)
            for i_outer in range(iter_outer):   
                if noise_app == True: 
                    thre_list = np.linspace(0, sigma2_app/100,iter_outer*iter1) 
                if i_outer == 0:
                    H_est = torch.from_numpy(H_thre_full_real[bs]).type(dtype).to(device)
                    H_bpic = H_thre_full_real[bs] 
                else:
                     
                    for ih in range(iter_NN1):    
                        net_input1 = z1
                        h_est = net1(net_input1)   
                         
         
                        o4_np = h_est.clone().detach().cpu().numpy()
                        lenh = (lmax+1)*(2*kmax+1)
                        o4_np_c = o4_np[0:lenh]+1j*o4_np[lenh:] 
                        
                        iter_h = ih + iter_NN1*(i_outer-1) 
                        thre = thre_list[iter_h]  
                             
                        indices_c = np.where(abs(o4_np_c)**2<thre)
                        indices_real = np.concatenate((indices_c[0],indices_c[0]+lenh))
                        h_est[indices_real] = 0   
                        phi_h = torch.matmul(phi_p_full_torch+phi_x_np_torch,h_est.unsqueeze(-1))   
                        l1_norm1 = 0.01*torch.sum(abs(h_est)) 
                        loss2 = mse(y_full_torch,phi_h)  + l1_norm1 
                        optimizer1.zero_grad()      
                        loss2.backward()  
                        optimizer1.step()       
                        h_est_np = h_est.detach().cpu().numpy()
                        h_est_torch = torch.from_numpy(h_est_np).type(dtype).to(device) 
          
                        dip_h_arr[bs,iter_NN1*i_outer+ih,:] = h_est_np 
                        NMSE_hest[bs,iter_NN1*i_outer+ih] =  np.square(abs(h_est_np.reshape(-1,1) - h_vec_real[bs])).sum() / (np.square(abs(h_vec_real[bs])).sum())
                  
                    H_est_full,H_est_full_real = Bulid_Heff(h_est_torch,lmax,kmax,N,M,torch.complex64,device)    
                    
                    H_full =  H_est_full_real.detach().cpu().numpy()      
                    H_est = torch.from_numpy(H_full.squeeze()).type(dtype).to(device)    
                    H_bpic = H_full
                for ix in range(iter_NN2): 
                    optimizer2.zero_grad()  
                    net_input2 = z2
                    x_est = net2(net_input2)*np.max(self.constellation)   
                    Hx =  torch.matmul(H_est,(x_est+xp_torch).unsqueeze(-1))     
                    loss1 = mse(y_full_torch,Hx)   
                    optimizer2.zero_grad()      
                    loss1.backward()  
                    optimizer2.step()
                    x_est_np = x_est.detach().cpu().numpy()  
                    dip_x_arr[bs,iter_NN2*i_outer+ix,:] = x_est_np
                    SER_dip[bs,iter_NN2*i_outer+ix] = self.ser(x_est_np, xd_real[bs].squeeze())
                   
                if noise_app == True:
                    sigma2_bpic = sigma2_app
                else:
                    sigma2_bpic = sigma2_d/2
                y_data_new_np =  y_full_real[bs,:].squeeze() - np.matmul(H_bpic,xp_real[bs,:].reshape(-1,1)).squeeze() 
                x_est,ser_bpic_iter,num_iter = self.BPIC(x_est_np, H_bpic, y_data_new_np, iter_bpic, sigma2_bpic,xd_real[bs])  
                
                SER_bpic[bs,iter_bpic*i_outer:iter_bpic*(i_outer+1)] = ser_bpic_iter    
                if noise_app == True:
                    noise = y_full_real[bs,:].squeeze() - np.matmul(H_bpic,(xp_real[bs,:].reshape(-1,1)+x_est)).squeeze()
                    sigma2_app = np.var(noise)  
                    noise_app_arr[bs,i_outer] = sigma2_app
                    
                x_bpic_array[bs,i_outer,:] = x_est.squeeze()
                x_est_torch = torch.from_numpy(x_est.squeeze()).type(dtype).to(device) 
                phi_x_est = self.Build_phi_new(lmax,kmax,M,N,x_est_torch,device)                          
                phi_x_np = phi_x_est.detach().cpu().numpy()
                phi_x_np_torch = torch.from_numpy(phi_x_np).type(dtype).to(device)  
                
        return NMSE_hest,SER_dip,SER_bpic,dip_h_arr,dip_x_arr,x_bpic_array,noise_app_arr
      
    def DIP_detect_iter(self,y_full_real,H_full_real,phi_p_full,h_vec_real,xd_real,xp_real,N,M,sigma2_d,iter_NN1,iter_bpic):
    
        dtype =  torch.float32
        device = torch.device('cuda', index=0)  
         
        SER_dip = np.zeros((self.batch_size,iter_NN1 ))  
        SER_bpic = np.zeros((self.batch_size,iter_bpic ))  
        dip_x_arr = np.zeros((self.batch_size,iter_NN1,xd_real.shape[-1]))    
        for bs in range(self.batch_size):       
            net2  = DD_det(xd_real.shape[-1]).type(dtype).to(device) 
            z = torch.randn(4).type(dtype).to(device)   
                 
            mse = torch.nn.MSELoss().type(dtype).to(device)  
            optimizer = torch.optim.Adam(net2.parameters(),lr=0.01) 
            
            y_full_torch = torch.from_numpy(y_full_real[bs].squeeze()).type(dtype).to(device).unsqueeze(-1)
            H_full_torch = torch.from_numpy(H_full_real[bs].squeeze()).type(dtype).to(device)   
            xp_torch = torch.from_numpy(xp_real[bs].squeeze()).type(dtype).to(device)    
            for j in range(iter_NN1):
                # step 1, update network:  
                optimizer.zero_grad()   
                x_est = net2(z)*np.max(self.constellation)    
                Hx =  torch.matmul(H_full_torch,(x_est.reshape(-1)+xp_torch).unsqueeze(-1))     
                loss2 = mse(y_full_torch,Hx)    
                loss2.backward()  
                optimizer.step()     
                x_est_np = x_est.detach().cpu().numpy().reshape(-1)  
                dip_x_arr[bs,j,:] = x_est_np
                SER_dip[bs,j] = self.ser(x_est_np, xd_real[bs].squeeze())    
            y_data_new_np =  y_full_real[bs,:].squeeze() - np.matmul(H_full_real[bs],xp_real[bs,:].reshape(-1,1)).squeeze()
          
            x_est,ser_bpic_iter,num_iter = self.BPIC(x_est_np, H_full_real[bs], y_data_new_np, iter_bpic, sigma2_d/2,xd_real[bs])    
  
            SER_bpic[bs,:] = ser_bpic_iter  
        return SER_dip,dip_x_arr,SER_bpic,x_est
 