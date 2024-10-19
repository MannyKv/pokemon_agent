from functools import cached_property

import numpy as np
from pyboy.utils import WindowEvent

from pyboy_environment.environments.pokemon.pokemon_environment import (
    PokemonEnvironment,
)
from pyboy_environment.environments.pokemon import pokemon_constants as pkc


class PokemonBrock(PokemonEnvironment):
    def __init__(
            self,
            act_freq: int,
            emulation_speed: int = 0,
            headless: bool = False,
    ) -> None:

        valid_actions: list[WindowEvent] = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            #  WindowEvent.PRESS_BUTTON_START,
        ]

        release_button: list[WindowEvent] = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            # WindowEvent.RELEASE_BUTTON_START,
        ]

        super().__init__(
            act_freq=act_freq,
            task="brock",
            init_name="has_pokedex.state",
            emulation_speed=emulation_speed,
            valid_actions=valid_actions,
            release_button=release_button,
            headless=headless,
        )

    # store the visited coordinates.
    visited_coords = []
    dictionary_visitations = {}
    same_location_counter = 0

    def _get_state(self) -> np.ndarray:
        # Implement your state retrieval logic here
        game_stats = self._generate_game_stats()
        isBattle = self._read_m(0xD057) != 0x00
        state_vector = [
            game_stats["location"]["x"],
            game_stats["location"]["y"],
            game_stats["location"]["map_id"],
            # game_stats["caught_pokemon"],
            # game_stats["seen_pokemon"]
            # game_stats["party_size"],
            # sum(game_stats["hp"]["current"]),  # total HP
            # game_stats["badges"],
            # game_stats["money"],
        ]

        return np.array(state_vector)

    def _calculate_reward(self, new_state: dict) -> float:
        # Implement your reward calculation logic here
        temp_reward = 0

        # check if new coordinate and reward.
        location = new_state.get("location")
        temp_reward += self.exploration_reward(location)
        # check to ensure bro is not revisiting same spots
       # temp_reward += self.overworld_movement_reward(new_state)
        # give a reward for if a pokemon is caught
        temp_reward += self._caught_reward(new_state) * 100

        temp_reward += self._seen_reward(new_state)

        temp_reward += self._using_the_right_move_in_battle_reward()
        # give a reward for battles indicated by pokemon level up
        temp_reward += self._levels_reward(new_state) * 100

        # check collision
        # temp_reward+=self.check_if_collided(self._get_screen_walkable_matrix())

        return temp_reward

    def _using_the_right_move_in_battle_reward(self) -> int:
        reward = 0

        # check if the agent is in battle
        battle = (self._read_m(0xD057) != 0x00)

        if battle:
            return 5
        return 0


    def exploration_reward(self, location):
        battle = (self._read_m(0xD057) != 0x00)
        if location["map_id"] not in self.visited_coords:
            self.visited_coords.append(location["map_id"])
            return 10

        if self.prior_game_stats["location"]["x"] != location["x"] or self.prior_game_stats["location"]["y"] != location["y"]:
            return 1
        elif not battle:
            return -0.01
        else:
            return 0


        # final_reward = 0
        # # Reward for visiting new (x, y) coordinates in a familiar map
        # if location not in self.visited_coords:
        #     final_reward += 1 * (1/self.steps)  # Increase reward for new areas
        #     self.visited_coords.append(location)
        #     if self.prior_game_stats["location"]["map_id"] != location["map_id"]:
        #         self.same_location_counter = 0
        # else:
        #     self.same_location_counter += 1
        #     index = self.visited_coords.index(location)
        #     final_reward -= 0.01 * (len(self.visited_coords) - index) # * by how long ago it was seen for more painful rewrds
        return 0

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        return game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here
        if self.same_location_counter > 4000:
            self.same_location_counter = 0
            return True
        # Maybe if we run out of pokeballs...? or a max step count
        return self.steps >= 1000
