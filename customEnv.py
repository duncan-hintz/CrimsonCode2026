from poke_env.environment.doubles_env import DoublesEnv

from typing import Optional, Union

import numpy as np
from gymnasium.spaces import Discrete

from poke_env.battle import AbstractBattle
from poke_env.battle.double_battle import DoubleBattle
from poke_env.environment.env import ObsType
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    DoubleBattleOrder,
    ForfeitBattleOrder,
    PassBattleOrder,
    SingleBattleOrder,
)
from poke_env.player.player import Player
from poke_env.ps_client import (
    AccountConfiguration,
    LocalhostServerConfiguration,
    ServerConfiguration,
)
from poke_env.teambuilder import Teambuilder

import webbrowser

from time import sleep

class CustomEnv(DoublesEnv):
    metadata={}

    def __init__(
        self,
        account_configuration1: Optional[AccountConfiguration] = None,
        account_configuration2: Optional[AccountConfiguration] = None,
        avatar: Optional[int] = None,
        battle_format: str = "gen9randomdoublesbattle",
        log_level: Optional[int] = None,
        save_replays: Union[bool, str] = False,
        server_configuration: Optional[
            ServerConfiguration
        ] = LocalhostServerConfiguration,
        accept_open_team_sheet: Optional[bool] = False,
        start_timer_on_battle_start: bool = False,
        start_listening: bool = True,
        open_timeout: Optional[float] = 10.0,
        ping_interval: Optional[float] = 20.0,
        ping_timeout: Optional[float] = 20.0,
        challenge_timeout: Optional[float] = 60.0,
        team: Optional[Union[str, Teambuilder]] = None,
        fake: bool = False,
        strict: bool = False,
        render_mode: Optional[str] = None,
    ):
        super().__init__(
            account_configuration1=account_configuration1,
            account_configuration2=account_configuration2,
            avatar=avatar,
            battle_format=battle_format,
            log_level=log_level,
            save_replays=save_replays,
            server_configuration=server_configuration,
            accept_open_team_sheet=accept_open_team_sheet,
            start_timer_on_battle_start=start_timer_on_battle_start,
            start_listening=start_listening,
            open_timeout=open_timeout,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
            challenge_timeout=challenge_timeout,
            team=team,
            fake=fake,
            strict=strict,
        )
        self.action_spaces = {
            agent: Discrete(107*107) for agent in self.possible_agents
        }
        self.render_browser_open=False
        self.render_mode=render_mode
    
    def reset(self,seed=None,options=None):
        self.render_browser_open = False
        return super().reset(seed=seed,options=options)

    def get_mask(self, battle: AbstractBattle):
        #Initial action masking for gen 9, removing other gimmicks
        action_mask=[0,]*107
        action_mask2=[0,]*107

        #orders is in the form of [[Orders],[Orders]] where each nested list is the valid orders for each slot
        #it does not account for invalid moves together (like double tera)
        for i in range(2):
            for order in battle.valid_orders[i]:
                orderNum=DoublesEnv._order_to_action_individual(order=order,battle=battle,fake=False,pos=i)
                if(i==0):
                    action_mask[orderNum]=1
                else:
                    action_mask2[orderNum]=1

        #DoubleBattleOrder.join_orders()

        #Converting the masks to the proper type
        action_mask=np.array(action_mask,dtype=np.int8)
        action_mask2=np.array(action_mask2,dtype=np.int8)
        #Combine into a single dimensional array of all combinations
        action_mask_combined=np.sum(np.array(np.meshgrid(action_mask, action_mask2)).T.reshape(-1, 2),axis=1)
        #If only one choice was valid, mask the combination
        action_mask_combined[action_mask_combined <2]=0
        action_mask_combined[action_mask_combined==2]=1
        #Now mask illegal combined operations:
        #To get combination index of [x,y], do: [107*x + y]
        #To undo: num//107 = x, num%107=y
        #Both pass:
        action_mask_combined[0]=0
        #Both switch to same target:
        for i in range(1,7):
            action_mask_combined[108*i]=0
        #Both tera:
        for i in range(87,107):
            for j in range(87,107):
                action_mask_combined[107*i + j]=0
        action_mask_combined=np.array(action_mask_combined,dtype=np.int8)
        return action_mask_combined

    
    def embed_battle(self, battle: AbstractBattle) -> tuple[ObsType,dict[int:int]]:
        """
        Returns the embedding of the current battle state in a format compatible with
        the Gymnasium API.

        :param battle: The current battle state.
        :type battle: DoubleBattle

        :return: The embedding of the current battle state.
        """
        assert isinstance(battle, DoubleBattle)

        if(battle.finished):
            return {"observations":None,"action_mask":None}

        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = np.zeros(8)
        moves_dmg_multiplier = np.ones(8)
        #For each available move per active pokemon on the team, 
        #embed base power and each damage multiplier

        for i, move in enumerate(battle.available_moves[0]):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if battle.opponent_active_pokemon[0] is not None:
                for active_pokemon in battle.opponent_active_pokemon:
                    if(active_pokemon !=None):
                        moves_dmg_multiplier[i] = move.type.damage_multiplier(
                            active_pokemon.type_1,
                            active_pokemon.type_2,
                            type_chart=active_pokemon._data.type_chart,
                        )
        for j, move in enumerate(battle.available_moves[1]):
            moves_base_power[3+j] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if battle.opponent_active_pokemon[0] is not None:
                for active_pokemon in battle.opponent_active_pokemon:
                    if(active_pokemon!=None):
                        moves_dmg_multiplier[3+j] = move.type.damage_multiplier(
                            active_pokemon.type_1,
                            active_pokemon.type_2,
                            type_chart=active_pokemon._data.type_chart,
                        )
        

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with n components
        final_vector = np.concatenate(
            [
                moves_base_power, #The eight available moves
                moves_dmg_multiplier, #For each available move, the damage multiplier against the active pokemon
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )

        action_mask=self.get_mask(battle)

        if self.render_mode == "human":
            self.render()

        return {"observations":np.float32(final_vector),"action_mask":action_mask}



    def calc_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=20.0, hp_value=10.0, victory_value=100.0,status_value=2.0
        )
    
    def step(self,actions):
        #Convert single nums to double
        actions={a:(actions[a]//107,actions[a]%107) for a in self.agents}
        return super().step(actions)
    
    def render(self):        
        if self.battle1 is not None:
            if(not self.render_browser_open):
                url = "https://localhost.psim.us/" + self.battle1.battle_tag
                webbrowser.open(url, new=0, autoraise=True)
                self.render_browser_open = True
                sleep(3)

            print(
                "  Turn %4d. | [%s][%3d/%3dhp] %10.10s, [%3d/%3dhp] %10.10s - %10.10s [%3d%%hp], %10.10s[%3d%%hp][%s]"
                % (
                    self.battle1.turn,
                    "".join(
                        [
                            "⦻" if mon.fainted else "●"
                            for mon in self.battle1.team.values()
                        ]
                    ),
                    0 if self.battle1.active_pokemon[0]==None else self.battle1.active_pokemon[0].current_hp,
                    0 if self.battle1.active_pokemon[0]==None else self.battle1.active_pokemon[0].max_hp,
                    "None" if self.battle1.active_pokemon[0]==None else self.battle1.active_pokemon[0].species,
                    0 if self.battle1.active_pokemon[1]==None else self.battle1.active_pokemon[1].current_hp,
                    0 if self.battle1.active_pokemon[1]==None else self.battle1.active_pokemon[1].max_hp,
                    "None" if self.battle1.active_pokemon[1]==None else self.battle1.active_pokemon[1].species,
                    "None" if self.battle1.opponent_active_pokemon[0]==None else self.battle1.opponent_active_pokemon[0].species,
                    0 if self.battle1.opponent_active_pokemon[0]==None else self.battle1.opponent_active_pokemon[0].current_hp,
                    "None" if self.battle1.opponent_active_pokemon[1]==None else self.battle1.opponent_active_pokemon[1].species,
                    0 if self.battle1.opponent_active_pokemon[1]==None else self.battle1.opponent_active_pokemon[1].current_hp,
                    "".join(
                        [
                            "⦻" if mon.fainted else "●"
                            for mon in self.battle1.opponent_team.values()
                        ]
                    ),
                ),
                end="\n" if self.battle1.finished else "\r",
            )