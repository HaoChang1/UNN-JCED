import numpy as np   
from OTFS_sys import *
from OTFS_ResGrid import *  

def otfs_data_gen(batch_size,N,M,lmax,kmax,p,SNR_d,SNR_p,Mod,pl_len,pk_len,thre_ce=False):   
    if Mod == 4:
        constel = [-0.7071-0.7071j, -0.7071+0.7071j, 0.7071-0.7071j, 0.7071+0.7071j];
    if Mod == 16:   
        constel = np.array([-0.948683298050514 + 0.948683298050514j, -0.948683298050514 + 0.316227766016838j, -0.948683298050514 - 0.948683298050514j, -0.948683298050514 - 0.316227766016838j, -0.316227766016838 + 0.948683298050514j, -0.316227766016838 + 0.316227766016838j, -0.316227766016838 - 0.948683298050514j, -0.316227766016838 - 0.316227766016838j, 0.948683298050514 + 0.948683298050514j, 0.948683298050514 + 0.316227766016838j, 0.948683298050514 - 0.948683298050514j, 0.948683298050514 - 0.316227766016838j, 0.316227766016838 + 0.948683298050514j, 0.316227766016838 + 0.316227766016838j, 0.316227766016838 - 0.948683298050514j, 0.316227766016838 - 0.316227766016838j]);
    
    guard_delay_num_neg=lmax
    guard_delay_num_pos=lmax
    guard_doppler_num_neg=2*kmax
    guard_doppler_num_pos=2*kmax  
    ##### for test embedded pilot
    # num_syms =   M*N-(guard_doppler_num_neg+guard_doppler_num_pos+pk_len)*(guard_delay_num_neg+guard_delay_num_pos+pl_len)
    # sigma2_d = 1/np.power(10,SNR_d/10)
    # pilot_pow = sigma2_d*np.power(10,SNR_p/10)
    # sigma2_p = 1/np.power(10,(SNR_p)/10)
    # data_pow = 1
     
    num_syms =   M*N
    sigma2_d = 1/np.power(10,SNR_d/10)
    data_pow = 1
    pilot_pow = sigma2_d*np.power(10,SNR_p/10)
    sigma2_p = 1/np.power(10,(SNR_p)/10) 
    
    xDD_idx = np.random.randint(Mod, size=(batch_size, num_syms));
    xDD = np.take(constel,xDD_idx)*data_pow;
    
    xp_idx = np.random.randint(Mod, size=(pl_len*pk_len)); 
    ##### Using random pilot
    xp = np.take(constel,xp_idx)*np.sqrt(pilot_pow) 
    rg = OTFSResGrid(M, N, batch_size=batch_size);
    # rg.setPulse2Ideal();
    rg.setPulse2Recta();
    rg.setPilot2Center(pl_len, pk_len);
    rg.setGuard(guard_delay_num_neg, guard_delay_num_pos,guard_doppler_num_neg,guard_doppler_num_pos)

    rg.setPilot2SuperImposed()
    rg.map(xDD,pilots=xp, pilots_pow=pilot_pow);
     
    X_all = rg.content
    X_p = rg.X_p
    X_d = X_all - X_p 
    otfs = OTFS(batch_size=batch_size);
    otfs.modulate(rg);  
       
    otfs.setChannel(p, lmax, kmax,isNorm=False,isTDL_E=True);  
    otfs.passChannel(sigma2_d);
    # otfs.passChannel(0);
    his, lis, kis = otfs.getCSI(); 
    
    H_full,row_ids,col_ids = otfs.getChannel(data_only=False);  
 
    phi_p_full,h_p = otfs.buildPhi_h(lmax, kmax,his, lis, kis,X_p)
    phi_all,h_all = otfs.buildPhi_h(lmax, kmax,his, lis, kis,X_all)
     
    h_vec = h_p.reshape(batch_size,-1,1,order='F') 
    x_p_vect = X_p.reshape(batch_size,-1)  
    rg_rx = otfs.demodulate(); 
    yDD_full = rg_rx.getContent(isVector=True);  
    h_vec_r = np.real(h_vec);
    h_vec_i = np.imag(h_vec);
    h_vec_real = np.concatenate([h_vec_r, h_vec_i], axis=1)   # (batch,168)
    
      
    H_full_r = np.real(H_full);
    H_full_i = np.imag(H_full);
    H_full_real = np.concatenate([np.concatenate([H_full_r, -H_full_i], axis=2), np.concatenate([H_full_i, H_full_r], axis=2)], axis=1) 
    
    phi_p_full_r = np.real(phi_p_full);
    phi_p_full_i = np.imag(phi_p_full);
    phi_p_full_real = np.concatenate([np.concatenate([phi_p_full_r, -phi_p_full_i], axis=2), np.concatenate([phi_p_full_i, phi_p_full_r], axis=2)], axis=1) 
    
    phi_all_r = np.real(phi_all);
    phi_all_i = np.imag(phi_all);
    phi_all_real = np.concatenate([np.concatenate([phi_all_r, -phi_all_i], axis=2), np.concatenate([phi_all_i, phi_all_r], axis=2)], axis=1)   
    
    omega2 = phi_all_real[0].squeeze() 
    a2 = np.transpose(omega2,(1,0))  @ omega2 
    b2 = a2/(sigma2_d/2)
    crlb2 = (1/b2[0,0])*2*p
    crlb_data = crlb2/np.sum(h_vec_real[0].squeeze()**2)  
 
    yfull_r = np.real(yDD_full);
    yfull_i = np.imag(yDD_full);
    y_full_real = np.concatenate([yfull_r, yfull_i], axis=1)    
  
    xd_r = np.real(xDD);
    xd_i = np.imag(xDD);
    xd_real = np.concatenate([xd_r, xd_i], axis=1)   
    xp_r = np.real(x_p_vect);
    xp_i = np.imag(x_p_vect);
    xp_real = np.concatenate([xp_r, xp_i], axis=1)    
    if thre_ce == True:  
        if int(pk_len*pl_len)==1:
            #####avoid the empty channel estimates by Threshold scheme
            thre3 = 3*np.sqrt(1+sigma2_d)  
            thre25 = 2.5*np.sqrt(1+sigma2_d)
            thre2 = 2*np.sqrt(1+sigma2_d)
            thre15 = 1.5*np.sqrt(1+sigma2_d)
            thre1 = np.sqrt(1+sigma2_d)
            thre0 = 0
             
            yDD_data, his_est, lis_est, kis_est = rg_rx.demap(threshold=thre3);  
     
            if len(his_est) == 0:
                his_est = np.zeros((batch_size,1))
            if np.sum(np.all(his_est == 0, axis=1))>0:
                _, his_est, lis_est, kis_est = rg_rx.demap(threshold=thre25); 
                
            if len(his_est) == 0:
                his_est = np.zeros((batch_size,1))
            if np.sum(np.all(his_est == 0, axis=1))>0:
                _, his_est, lis_est, kis_est = rg_rx.demap(threshold=thre2); 
                
            if len(his_est) == 0:
                his_est = np.zeros((batch_size,1))
            if np.sum(np.all(his_est == 0, axis=1))>0:
                _, his_est, lis_est, kis_est = rg_rx.demap(threshold=thre15);
                
            if len(his_est) == 0:
                his_est = np.zeros((batch_size,1))  
            if np.sum(np.all(his_est == 0, axis=1))>0:
                _, his_est, lis_est, kis_est = rg_rx.demap(threshold=thre1);  
                
            if len(his_est) == 0:
                his_est = np.zeros((batch_size,1)) 
            if np.sum(np.all(his_est == 0, axis=1))>0:
                _, his_est, lis_est, kis_est = rg_rx.demap(threshold=thre0);   
            
            h_thre = np.zeros((batch_size,2*kmax+1,lmax+1), dtype=complex)
            for bb in range(batch_size):
                if len(his_est) != 0 :                
                    for indx in range(his_est.shape[1]):
                        k_ind = kis_est+kmax
                        h_thre[bb,k_ind[bb,indx],lis_est[bb,indx]] = his_est[bb,indx]
     
            h_thre_vec = h_thre.reshape(batch_size,-1,1,order='F')  
            h_thre_r = np.real(h_thre_vec);
            h_thre_i = np.imag(h_thre_vec);
            h_thre_real = np.concatenate([h_thre_r, h_thre_i], axis=1)  
            NMSE_thre = np.square(h_vec_real - h_thre_real).sum() / (np.square(h_vec_real).sum())   
            
            try: 
                H_est_full,_,_ = otfs.getChannel(his_est, lis_est, kis_est, data_only=False);
            except: 
                H_est_full = np.zeros(H_full.shape)
            
            H_est_full_r = np.real(H_est_full);
            H_est_full_i = np.imag(H_est_full);
            H_est_full_real = np.concatenate([np.concatenate([H_est_full_r, -H_est_full_i], axis=2), np.concatenate([H_est_full_i, H_est_full_r], axis=2)], axis=1) 
    else:
        H_est_full_real = 0  
        h_thre_real = 0
        NMSE_thre = 0
     
    return H_full,H_full_real,h_vec_real,phi_p_full_real,phi_all_real,y_full_real,xd_real,X_d,xp_real,sigma2_d,sigma2_p,NMSE_thre,H_est_full_real,h_thre_real,crlb_data
         
