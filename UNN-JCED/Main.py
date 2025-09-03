import numpy as np  
import matplotlib.pyplot as plt  
import time    
from DIP_joint_iter_all import DIP_joint 
from OTFS_Data_Gen import otfs_data_gen
import torch



num_batch = 1
batch_size = 2
 
 
np.random.seed(14)
torch.manual_seed(14)

M=16
N=16
lmax=3
kmax=3
p=8
 
pl_len = 1
pk_len = 1
  
Mod = 4 
LR = 0.01

iter_outer = 20
iter_bpic = 10
iter_NN1 = iter_NN2 = 20
start = time.time()
 
# SNR_p = 28
# SNR_d = 10  
arr_snr_d = np.array([4,6,8,10 ])

plot_nmse_thre = []     
plot_nmse_joint = []      
plot_nmse_joint1 = []  
plot_nmse_crlb = []  
plot_ser_joint = []     
plot_ser_joint1 = []    
plot_ser_csi = []    
  
  
for SNR_d in arr_snr_d:  
    SNR_p = SNR_d + 20
    list_nmse_thre = []    
    list_nmse_joint = []     
    list_nmse_joint1 = []  
    list_nmse_crlb = []     
    list_ser_joint = []      
    list_ser_joint1 = []      
    list_ser_csi = []        
     
    for batch in range(num_batch):
        H_full,H_full_real,h_vec_real,phi_p_full_real,phi_all_real,y_full_real,xd_real,X_d,xp_real,sigma2_d,sigma2_p,NMSE_thre,H_thre_full_real,h_thre_real,crlb = otfs_data_gen(batch_size, N, M, lmax, kmax, p, SNR_d, SNR_p, Mod, pl_len,pk_len,thre_ce=True)    
        list_nmse_thre.append(NMSE_thre)  
        list_nmse_crlb.append(crlb)     
              
        dip = DIP_joint(Mod, LR, batch_size)
         
        NMSE_joint,SER_dip_joint,SER_bpic_joint,dip_h_arr,dip_x_arr,x_bpic_array,_ = dip.DIP_joint_alter_new(y_full_real,phi_p_full_real,H_thre_full_real,h_thre_real,h_vec_real,xd_real,xp_real,sigma2_d,sigma2_p,lmax,kmax,N,M,iter_outer,iter_NN1,iter_NN2,iter_bpic,noise_app=False)
        NMSE_joint1,SER_dip_joint1,SER_bpic_joint1,dip_h_arr1,dip_x_arr1,x_bpic_array1,_ = dip.DIP_joint_alter_new(y_full_real,phi_p_full_real,H_thre_full_real,h_thre_real,h_vec_real,xd_real,xp_real,sigma2_d,sigma2_p,lmax,kmax,N,M,iter_outer,iter_NN1,iter_NN2,iter_bpic,noise_app=True)
        
        SER_dip_csi,_,SER_bpic_csi,_ = dip.DIP_detect_iter(y_full_real,H_full_real,phi_p_full_real,h_vec_real,xd_real,xp_real,N,M,sigma2_d,iter_outer*iter_NN1,iter_bpic) 
           
        list_nmse_joint.append(np.mean(NMSE_joint[:,-1]))  
        list_nmse_joint1.append(np.mean(NMSE_joint1[:,-1]))    
        list_ser_joint.append(np.mean(SER_bpic_joint[:,-1]))   
        list_ser_joint1.append(np.mean(SER_bpic_joint1[:,-1])) 
        list_ser_csi.append(np.mean(SER_bpic_csi[:,-1]))     
           
    plot_nmse_thre.append(np.mean(list_nmse_thre))    
    plot_nmse_joint.append(np.mean(list_nmse_joint))    
    plot_nmse_joint1.append(np.mean(list_nmse_joint1)) 
    plot_nmse_crlb.append(np.mean(list_nmse_crlb))      
    
    plot_ser_joint.append(np.mean(list_ser_joint))    
    plot_ser_joint1.append(np.mean(list_ser_joint1))  
    plot_ser_csi.append(np.mean(list_ser_csi))       
      
end = time.time()
print('run time:%0.2f' %(end-start))    
 
plt.plot(arr_snr_d,plot_nmse_thre,label=' Threshold')     
plt.plot(arr_snr_d,plot_nmse_joint,label=' UNN-JCED')      
plt.plot(arr_snr_d,plot_nmse_joint1,label=' UNN-JCED-noise-app') 
plt.plot(arr_snr_d,plot_nmse_crlb,label=' CRLB')              
plt.xlabel('SNR_d')
plt.ylabel('NMSE')  
plt.semilogy()
plt.legend() 
plt.show()
    
 
plt.plot(arr_snr_d,plot_ser_joint,label=' UNN-JCED')   
plt.plot(arr_snr_d,plot_ser_joint1,label=' UNN-JCED-noise-app')   
plt.plot(arr_snr_d,plot_ser_csi,label=' CSI-UNN-BPIC')     
plt.xlabel('SNR_d')
plt.ylabel('SER')  
plt.semilogy()
plt.legend() 
plt.show()

   























