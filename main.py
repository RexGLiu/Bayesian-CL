import torch

# from modules.SAC import SAC
from modules.PPO import PPO
from modules.model import Target, Controller
from modules.target_env import TargetNetEnv

def copy_networks(copy_source_net, copy_target_net):
    for source_param, target_param in zip(copy_source_net.parameters(), copy_target_net.parameters()):
        target_param.data.copy_(source_param.data)

def main():
    device = torch.device('cuda') if torch.cuda.is_available() \
        else torch.device('mps') if torch.backends.mps.is_available() \
        else torch.device('cpu')

    device = torch.device('cpu')

    # create target network
    input_dim, target_hidden, output_dim = 10, 3, 5
    target_net = Target(input_dim=input_dim, output_dim=output_dim, hidden_dim=target_hidden, device=device)
    env = TargetNetEnv(target_net, device)

    eval_net = Target(input_dim=input_dim, output_dim=output_dim, hidden_dim=target_hidden, device=device)
    copy_networks(target_net, eval_net)
    eval_env = TargetNetEnv(eval_net, device)

    # create controller network
    controller_hidden = 256
    controller_input = output_dim + target_hidden + 2  # input for PPO excludes entropy
    n_policy_hidden = 2
        # LSTM receives inputs of form (x, y, y_targ, action, loss, action_size, action_entropy)
    controller = Controller(input_dim=controller_input, hidden_dim=controller_hidden, n_actions=target_hidden, n_policy_hidden=n_policy_hidden, device=device)

    # SAC_agent = SAC(env, controller, device)
    # train_losses, target_losses = SAC_agent.train()

    PPO_agent = PPO(env, eval_env, controller, controller_input, device)
    policy_losses, value_loss, target_losses = PPO_agent.train()

    # normalize y-outputs to ensure all "tasks" on same scale






    # to do: modify to pin different output nodes
    # scenario 1: class incremental learning
    # scenario 2: same output nodes, input-output maps (over diff or possibly even same support)
    # possible extension: update controller periodically by randomly and independently picking gates in target network, and have controller learn to infer correct gates
    # need to assess forward & backward transfer (latter generalises catastrophic forgetting metric)
    # note that quality of task separation/overlap may encourage transfer or Stroop-induced catastrophic interference
        # can we argue that humans will also exhibit inteference in some scenarios? e.g. if learn to move right when seeing left arrow, then must re-learn to move right when seeing right arrow
    # consider also classification reversal tasks: e.g. swap class labels for same set of inputs
    # for performance metric: track target network performance as controller adapts (should be poor performance initially that improves, hopefully to better than untrained network)
    # can requisite adaptation time be shortened w controller updates?

if __name__ == "__main__":
    main()

