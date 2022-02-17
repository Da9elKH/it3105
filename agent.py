from environments.cart_pole import CartPole
from environments.tower_of_hanoi import TowerOfHanoi
from environments.the_gambler import TheGambler
from environments.environment import ProblemEnvironment
from critics.critic import Critic
from critics.nn import NeuralNetworkCritic
from utils.decaying_variable import DecayingVariable
from critics.table import TableCritic
from actor import Actor
from enum import Enum
import random


class GameType(Enum):
    CART_POLE = 1
    TOWER_OF_HANOI = 2
    GAMBLER = 3


class CriticType(Enum):
    TABLE = 1
    NEURAL_NETWORK = 2


class Agent:
    def __init__(self, env: ProblemEnvironment, act: Actor, crt: Critic):
        self.env = env
        self.act = act
        self.crt = crt

    def actor_critic_model(self, episodes: int):
        """ This functions runs the general actor critic model """

        # Initializing small state-values for CRITIC is done when accessing the value
        # Initializing 0 as SAP-value for ACTOR is done when accessing the value

        for episode in range(episodes):
            # Reset eligibilities for actor and critic
            # e(s,a) ← 0: e(s) ← 0 ∀s, a
            self.crt.clear()
            self.act.clear()

            # Initialize environment
            # s ← s_init
            state, valid_actions = self.env.reset()

            # Helpers for replay logic
            last_episode = episode == (episodes - 1)
            i = 0

            # Store data for later visualisation
            self.env.store_training_metadata(current_episode=episode, last_episode=last_episode, current_step=i, state=state, reinforcement=0)

            while not self.env.is_finished():
                # Keep track of the current state-step
                i += 1

                # ACTOR: Get next action
                # a′ ← Π(s′)
                action = self.act.next_action(state, valid_actions)

                # ENVIRONMENT: Do next action and receive reinforcement, save state in list
                from_state, action, reinforcement, state, valid_actions, terminal = self.env.step(action)

                # ACTOR: Get next action and update eligibility
                # e(s,a) ← 1
                self.act.set_eligibility(from_state, action)

                # CRITIC: Calculate TD-error and update eligibility
                # δ ← r+γV(s′)−V(s)
                td_error = self.crt.td_error(reinforcement, from_state, state, terminal)
                # e(s) ← 1
                self.crt.set_eligibility(from_state)

                # Adjustments for all state-action-pairs
                # ∀(s,a):
                #   V(s) ← V(s)+ α*δ*e(s)
                #   e(s) ← γλe(s)
                #   Π(s,a) ← Π(s,a)+α*δ*e(s,a)
                #   e(s,a) ← γλe(s,a)
                self.crt.adjust(td_error)
                self.act.adjust(td_error)

                # Store data for later visualisation
                self.env.store_training_metadata(current_episode=episode, last_episode=last_episode, current_step=i, state=state, reinforcement=reinforcement)

            # For batch learning if using neural network
            loss = self.crt.learn()
            print(f"Episode: {episode}, steps: {i}")

        self.env.replay(saps=self.act.get_saps(), values=self.crt.get_values())


if __name__ == '__main__':
    game = GameType.CART_POLE
    critic = CriticType.TABLE

    #print(random.getstate())

    if game == GameType.CART_POLE:
        environment = CartPole(pole_length=0.5, pole_mass=0.1, gravity=-9.8, time_step=0.02, buckets=(5, 5, 5, 5), view_update_rate=0.02, time_out=300)

        if critic == critic.NEURAL_NETWORK:
            n_episodes = 300
            critic = NeuralNetworkCritic(
                discount_factor=0.9,
                learning_rate=0.003,
                input_size=environment.input_space(),
                hidden_size=(32, 32)
            )
        else:
            # random.setstate((3, (2147483648, 3145687911, 1338532607, 2329591376, 42354391, 3446885120, 4213192133, 1614424256, 1369782798, 1359727412, 2044711073, 441931229, 234036136, 2131434818, 422488983, 2255683052, 3883231956, 3112632526, 3784155561, 1994242012, 4247747385, 1228017790, 3952315494, 1854213845, 1366124338, 2913029983, 3116978008, 197030663, 2095487681, 874493553, 3367749027, 2179162193, 1365151899, 3754862010, 368706091, 1391521898, 1948289569, 162672880, 2661055401, 3907484005, 3137704099, 244755977, 1007494454, 1525338309, 903579015, 1917599405, 2155298872, 2410325185, 593000753, 3216984363, 2520102969, 322136203, 1044921459, 1754135933, 1715210736, 1573994047, 3182185326, 2610521608, 657800561, 154780685, 151109484, 4069452163, 1650781253, 1264498866, 985454837, 2182390103, 2041434331, 819927280, 1126298770, 1646971723, 1578237863, 1116752305, 2922176709, 1279370981, 2734381055, 3469926655, 725585369, 1391915530, 1579386166, 1168830268, 315697443, 2641864082, 3052673497, 735074913, 2617494860, 1496189350, 3287306365, 1427708037, 2875375266, 4196397680, 2399578606, 3615152560, 486828705, 75812627, 2505833053, 2651199149, 4038083999, 2447050070, 4120428578, 3371856163, 1889379504, 979377597, 2306208351, 3691530986, 1212509661, 3827764278, 2101937516, 1818276522, 2482568774, 819763839, 1199323364, 3569571330, 433121525, 431429407, 3817522830, 2492529072, 3487363251, 3667293411, 787850172, 3746320197, 3391855622, 4052375553, 2225457971, 2809709275, 1663161378, 414641020, 50592768, 2642134133, 4161665005, 2656898850, 4009113279, 3811158592, 1491313574, 1776418924, 3085962690, 2320383231, 2485294541, 583035540, 3940311487, 1414129732, 689809534, 2983836927, 884606827, 2787028333, 915133480, 489995720, 802544674, 4287340850, 205902514, 1344454122, 3912372162, 2645789385, 1395765103, 3746358913, 2926324350, 4127135058, 1629418538, 461601925, 1009177110, 2802265145, 2180461260, 2011775059, 3704669494, 1846185384, 1217077909, 3814827663, 2909053358, 3383854323, 824368213, 1107375249, 453976723, 1542142438, 2733817733, 2454266343, 4057553102, 463751502, 3930396209, 2289344783, 3324287200, 3513069555, 1611112555, 3709864006, 1540094696, 2326334561, 3744062298, 1023147335, 237463909, 2518995706, 1840838943, 1358842658, 3621771855, 244639098, 1568910673, 4276007704, 95279724, 4248105095, 4052828006, 3883378704, 229934000, 1927755335, 1181009065, 414024382, 2094459628, 3495626811, 3457834394, 971028578, 2982053544, 3889041134, 95284953, 4153048174, 3182788508, 194117245, 3435071414, 1281875647, 3963452494, 4271184945, 1886543434, 4134642763, 2338797351, 3949536530, 2536687888, 2668521970, 1255566391, 88516119, 3809819585, 830602314, 2588847213, 726045064, 156895430, 2686521963, 2263182123, 1323933556, 3085168175, 1643434258, 135527832, 1899459446, 2594482098, 1448919403, 2433912981, 3353174757, 2100508820, 298614631, 2744935905, 417689111, 182601449, 2775871995, 2650455744, 148502646, 1776527945, 3901273688, 806801798, 577992399, 3825632722, 1804641936, 396448122, 1189237940, 2754981140, 2875027669, 1038400835, 3368627444, 1550626230, 932495205, 126256042, 387253472, 3691378708, 2362638792, 3959075438, 122481420, 1133111252, 2866495376, 347346144, 3661050355, 414325780, 614024588, 3822112276, 1253680395, 1726415673, 2123134455, 2453819333, 1647834147, 195778250, 301066380, 3094101414, 2971816150, 374814016, 836606071, 4213606053, 3456737063, 3211797068, 187822956, 620704640, 1855022131, 4291071737, 90951292, 1465607044, 3300863573, 4069205184, 657268383, 4236835008, 1202442583, 2752389524, 1509413927, 3024732565, 1944081699, 1128696019, 230275598, 3812212125, 415617710, 816402143, 768705356, 1311866988, 3668012594, 3352772945, 1257958374, 1946815887, 3442153252, 3216491453, 2307103412, 108592066, 3117495405, 5139020, 761208909, 226033114, 515671151, 1480029124, 3108688401, 495621977, 3516117228, 323225779, 1756147263, 3175991217, 693495220, 290301947, 2616116225, 3087132368, 981806168, 2809780488, 1989964780, 1491096341, 1628547311, 731026276, 933482736, 2799613664, 56540117, 302734083, 1573520935, 3579884541, 1455357595, 2676516678, 4133979607, 2435733953, 2168633199, 4021940981, 3966257581, 3341652213, 2062208000, 2897999259, 425395047, 2335930206, 288386735, 1693704924, 2532458023, 1039004609, 2560077249, 1948691521, 2761149196, 4038112104, 2662258083, 1286247595, 3122870475, 3872528221, 2048859705, 757090227, 4115164154, 1464743146, 4260220908, 2461700462, 953294213, 2315859383, 2062045736, 1127301195, 1883287957, 1479421644, 2811600470, 510388098, 306645500, 3786736300, 335140728, 366003498, 929062034, 4285227823, 3455896509, 1382940782, 130409745, 746820581, 3580819111, 102432112, 383190110, 1178523850, 3106069062, 3148494121, 2260745687, 4111224780, 2373017008, 2060037309, 2825268404, 2226380085, 4156991829, 1550746910, 1612722886, 2606029966, 3746653287, 67780414, 1150848478, 904512903, 1595152671, 108506254, 3174014787, 1665471322, 4258707357, 1646962052, 560218258, 3576932605, 4259222567, 995028484, 2721859432, 3482175412, 4209726654, 1771249337, 748814696, 2950941441, 3742028251, 1371024253, 2553132901, 2913228905, 490146885, 3974580291, 2423228547, 86158833, 516771499, 2105633369, 3392153642, 1302455800, 3052151899, 3021856033, 1854954261, 3837872318, 1143213819, 3131014822, 1081327345, 2130993675, 2520089153, 3379965137, 3204566509, 608403868, 1411747141, 1026087761, 1730158395, 2220431112, 737251098, 3553390042, 3311513720, 1574268956, 4050233642, 3038670104, 2342257171, 2211444389, 1545088269, 2996650318, 233492880, 2653178543, 1357134713, 3275667789, 4122328598, 1813785154, 3725839868, 4281203320, 1216792802, 3033328251, 286106100, 923099242, 25908609, 669112208, 2048355756, 3386943099, 10553966, 3553133445, 2761433304, 751669699, 1294134816, 4221132893, 4242844311, 453770049, 2277481403, 1730141824, 1915451401, 4175844086, 2714705867, 2919425463, 2984799884, 1945034634, 3697494228, 448225961, 3971204822, 3578831328, 442736486, 2413351237, 564847501, 4145428560, 2263818767, 598029943, 3423124919, 3315017215, 1860947982, 3191779864, 3128567410, 2316348560, 3640806057, 990643138, 360721361, 688355174, 2524897969, 1967624888, 3231801295, 1173545968, 828832206, 3714833779, 1185527977, 1236898548, 2182594951, 2090321494, 1536758083, 2650381948, 1220314990, 662726812, 3030988093, 2903970400, 1501885240, 2561619745, 3137480018, 1420018910, 3156275598, 2907357096, 1716618016, 3279650346, 737118790, 3208417564, 156331218, 437603930, 3359399860, 3180887882, 482795322, 633952416, 1457874913, 2047790060, 3078911916, 969755491, 1527830264, 1061297010, 3943423233, 3077223540, 349806207, 1079569363, 3235208947, 2790924782, 784985699, 1892587698, 1219140745, 2448024329, 1416642048, 2308812224, 2275477061, 3719881031, 3415562637, 1440028629, 3829778233, 2069917670, 1347124476, 479315078, 2288299060, 2666557546, 3713661497, 2401971826, 881939160, 365050952, 4149787087, 3171049421, 1816815672, 4059670603, 1174533501, 785383803, 2026930571, 3426783876, 1674374228, 584605270, 240592525, 94653636, 2486033036, 3552776824, 2776345625, 605134780, 2490303833, 789280290, 2326219945, 3315935092, 165820551, 3415988811, 372684131, 555943423, 2420658976, 256783107, 2266141301, 4053798314, 3210755431, 3441256777, 3310240419, 45548403, 1758070114, 4137265789, 4243304199, 1869905739, 694343000, 1784541161, 2800833925, 624), None))
            n_episodes = 500
            critic = TableCritic(
                discount_factor=0.9,
                trace_decay=0.9,
                learning_rate=0.2
            )

        actor = Actor(
            discount_factor=0.9,
            trace_decay=0.9,
            learning_rate=0.2,
            epsilon=DecayingVariable(
                start=1.0,
                end=0.005,
                linear=True,
                episodes=n_episodes,
                episodes_end_value=0
            ),
        )

    elif game == GameType.TOWER_OF_HANOI:
        n_episodes = 500
        environment = TowerOfHanoi(num_pegs=3, num_discs=4, view_update_rate=0.5, time_out=300)
        actor = Actor(
            discount_factor=0.9,
            trace_decay=0.6,
            learning_rate=0.01,
            epsilon=DecayingVariable(
                start=1.0,
                end=0.001,
                episodes=n_episodes-25,
                linear=False
            )
        )

        if critic == CriticType.NEURAL_NETWORK:
            critic = NeuralNetworkCritic(
                discount_factor=0.95,
                learning_rate=0.003,
                input_size=environment.input_space(),
                hidden_size=(64, 32)
            )
        else:
            critic = TableCritic(
                discount_factor=0.9,
                trace_decay=0.6,
                learning_rate=0.05
        )

    else:
        environment = TheGambler(win_probability=0.4, state_space=100, time_out=300)
        if critic == CriticType.NEURAL_NETWORK:
            n_episodes = 1000
            critic = NeuralNetworkCritic(
                discount_factor=1,
                learning_rate=0.003,
                input_size=environment.input_space(),
                hidden_size=(64, 32, 16)
            )
            actor = Actor(
                discount_factor=1,
                trace_decay=0.2,
                learning_rate=0.2,
                epsilon=DecayingVariable(
                    start=0.5,
                    end=0.001,
                    episodes=n_episodes,
                    linear=False
                ),
            )
        else:
            n_episodes = 100000
            critic = TableCritic(
                discount_factor=1,
                trace_decay=0.8,
                learning_rate=0.03
            )
            actor = Actor(
                discount_factor=1,
                trace_decay=0.8,
                learning_rate=0.01,
                epsilon=DecayingVariable(
                    start=1,
                    end=0.001,
                    episodes=n_episodes,
                    linear=False
                ),
            )

    agent = Agent(env=environment, act=actor, crt=critic)
    agent.actor_critic_model(n_episodes)
