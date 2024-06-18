from module.actor_critic import *
from module.q_learning import *
import pickle
#--------------------------------------------------------------------------------------------------------------------------------

# set up streamlit interface layout
sl.set_page_config(layout="wide")

# model loading as cached data
@sl.cache_resource()
def loadPretrainedModel():
    # load pretrained actor-critic model
    model = MNISTNet()
    model.load_state_dict(torch.load('test_out.pt').state_dict())
    
    # load Actor-Critic class and set classification model to 'model'
    reloaded_agent = ActorCriticNNAgent(MNISTNet, obs_to_input=obs_to_input, df=0.1)
    reloaded_agent.model = model
    reloaded_agent.trainable = False
    
    # load Q Learning model
    with open('./qlearn_policy.pkl', 'rb') as f:
        loaded_policy = pickle.load(f)
        
    q_learning_model = q_learning(loaded_policy)
    
    return reloaded_agent, q_learning_model

a2c_model, qlearning_model = loadPretrainedModel()
  
#--------------------------------------------------------------------------------------------------------------------------------
def init_col1(col1):
    """This function defines the intitial appearance of column in the left in streamlit interface"""
    
    # configurating column 1
    with col1:
        # selection box for model selection
        sl.header("Model")
        model_select = sl.selectbox('Choose your detection model',
                                    ('Advantage Actor-Critic',
                                     'Q-Learning'),
                                    index = None,
                                    placeholder='Select your model...',)
        
        sl.write('You selected', model_select)
        
        sl.write('---')
        
        text_input = sl.text_input("Random seed configuration",
                                   placeholder="Enter your random seed here")
        
        # classify buttons
        detect_button = sl.button("Sample run", type="primary")
        
        return model_select, detect_button, text_input
    
def init_col2(col2):
    """This function defines the intitial appearance of column in the right in streamlit interface"""
    
    # configurating column 2
    with col2:
        # Introduction
        sl.header("Image Classification Using Advantage Actor-Critic")
        
        sl.write('---')
        sl.subheader("Sample run section")
        

#------------------------------------------------------------------------------------------------------------------------

def sample_run(col, button, seed=7):
    if button:
        with col:
            env = MNISTEnv(type='test', seed=seed)
            obs = env.reset()
            done = False
            total_reward = 0

            while not done:
                env.render()
                action = env.action_space.sample()
                dir, Y_pred = action % 4, action // 4
                sl.write("Agent moved %s" % (['North', 'South', 'East', 'West'][dir]))
                sl.write("Agent guessed %d" % Y_pred)
                
                _, reward, done, _ = env.step(action)
                total_reward += reward
                sl.write("Received reward %.1f on step %d" % (reward, env.steps))
                sl.write("Current reward %.1f on step %d" % (total_reward, env.steps))
            
            env.render()
            sl.write('Run ended!!')
            sl.write(f"Final reward {total_reward}")
                
    
#------------------------------------------------------------------------------------------------------------------------

def main():
    # initiate 2 columns in streamlit UI
    col1, col2 = sl.columns([0.3, 0.7])
    
    # create the two columns in streamlit interface
    model, detect_button, seed = init_col1(col1)
    init_col2(col2)
    
    try: seed = int(seed)
    except: seed = None
    
    # do classification
    if model == 'Advantage Actor-Critic': sample_run(col2, detect_button, seed)
    else: 
        with col2: qlearning_model.test()

if __name__ == '__main__':
    main()    