import numpy as np
import matplotlib.pyplot as plt

light_orange = (248/255,200/255,173/255)
dark_orange = (249/255,95/255,7/255)

gamma = 1.0
r = str(gamma)

sims = ['./old/low_friction/logs/PPO2_1/','./old/normal_friction/logs/PPO2_10/', './transfer/logs/PPO2_1/']
names = ['low friction', 'normal friction', 'transfer']

def smooth(x,window_len=11,window='hanning'):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def plot_measures():
    window = 500
    ini_plot=2000
    fig = plt.figure()
    for i in range(0,len(sims)):
        # print(f'gamma={r}')
        # i = lbl_gammas[r]        
        # read from files
        episode_reward = np.load(f'{sims[i]}/dqn_episode_reward.npy')        
        # smoothed episode reward
        idx = 0
        timesteps = episode_reward[:,0]        
        reward = episode_reward[:,1] 
        time = list(timesteps) 
        rew = list(reward) 
        y_smooth = smooth(np.array(rew),window,'flat')
        reward_smooth = list(y_smooth[0:len(time)])
        ax1=plt.subplot(1, 1, 1)        
        ax1.plot(time[ini_plot:], reward_smooth[ini_plot:], label=f'{names[i]}')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Episode reward')
        ax1.title.set_text(f'PPO Humanoid')
        ax1.legend(loc="lower right", ncol=10)

        # smoothed loss
        # idx=0
        # floss = np.load(f'./output/DQN_{i}/dqn_loss.npy')        
        # timesteps = floss[:,0]        
        # loss1 = floss[:,1]               
        # time = list(timesteps) 
        # loss = list(loss1)
        # y_smooth = smooth(np.array(loss),window,'flat')
        # loss_smooth = list(y_smooth[0:len(time)])
        # ax2=plt.subplot(2, 1, 2)        
        # ax2.plot(time[ini_plot:], loss_smooth[ini_plot:], label=f'sim {i}')
        # ax2.set_xlabel('Time')
        # ax2.set_ylabel('Loss')

        # box = ax2.get_position()
        # ax2.set_position([box.x0, box.y0+box.height*0.05,
        #          box.width, box.height * 0.65])
        # ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.30),
        #   fancybox=True, shadow=True, ncol=10)
        # ax2.legend(loc="upper right", ncol=10)

        # # smoothed td error
        # idx = 0
        # f_tderror = np.load(f'./sim{gamma}/DQN_{i}/alpha_td_error.npy')                
        # timesteps = f_tderror[:,0]        
        # td1 = f_tderror[:,1]        
        # time = list(timesteps)
        # td = list(td1)
        # y_smooth = smooth(np.array(td),window,'flat')
        # td_smooth = list(y_smooth[0:len(time)])
        # ax3=plt.subplot(3, 1, 3)        
        # ax3.plot(time[ini_plot:], td_smooth[ini_plot:], label=f'sim {i}')
        # ax3.set_xlabel('Time')
        # ax3.set_ylabel('TD error')
        # ax3.legend()

    plt.show()

def plot_value_s0():
    window = 2000
    ini_plot = 800
    for i in range(1,2):      
        fvalue1 = np.load(f'./output/values_s0_iter{i}.npy')     
        value = fvalue1  
        y_smooth = smooth(np.array(value),window,'flat')
        ax1=plt.gca()
        ax1.plot(y_smooth[ini_plot:], label=f'sim {i}')
        ax1.legend()
    plt.show()


plot_measures()
# plot_value_s0()
