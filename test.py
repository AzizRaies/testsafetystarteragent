import gym
import open_safety_gym

from safe_rl import cpo




#all_envs = gym.envs.registry
#env = gym.make('SafetyHopperRun-v0')
#env = gym.make('SafetyHumanoidCircle-v0')
#env = gym.make('SafetyAntCircle-v0') #epcost=0
env = gym.make('SafetyBallReach-v2')



cpo(env_fn = lambda : env,epochs=150,cost_lim=25,ac_kwargs=dict(hidden_sizes=(64, 32)))
#trpo_lagrangian(env_fn = lambda : env,epochs=20,ac_kwargs=dict(hidden_sizes=(64, 32)))



#X_cpo = pickle.load(open('CPOAgentaverage_returns_and_Costs_per_epochSafetyBallCircle-v0.sav', 'rb'))
#X_trpo = pickle.load(open('TRPOAgent_average_returns_and_Costs_per_epochSafetyBallCircle-v0.sav', 'rb'))
#
#
#plt.figure(1)
#plt.plot(X_cpo[0],label='cpo')

#plt.plot(X_trpo[0],label='trpo_lagrangian')
#plt.legend()
#plt.savefig('EpRetSafetyBallCircle_v2')

#plt.figure(2)

#plt.plot(X_cpo[1],label='cpo')

#plt.plot(X_trpo[1],label='trpo_lagrangian')
#plt.legend()
#plt.savefig('SafetyBallCircle-v2')

#plt.figure(4)

#plt.plot(X_cpo[1],label='EpCost')


#plt.legend()
#plt.savefig('CPOAgent_average_Costs_per_epochSafetyBallCircle-v0')
