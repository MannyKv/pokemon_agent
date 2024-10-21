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
    seen = []
    # same_location_counter = 0
    action = -1
    left_wall = -1
    right_wall = -1
    top_wall = -1
    bottom_wall = -1

    def _run_action_on_emulator(self, action_array):
        action = action_array[0]
        action = min(action, 0.99)

        # Continuous Action is a float between 0 - 1 from Value based methods
        # We need to convert this to an action that the emulator can understand
        bins = np.linspace(0, 1, len(self.valid_actions) + 1)
        button = np.digitize(action, bins) - 1
        self.action = button

        # Push the button for a few frames
        self.pyboy.send_input(self.valid_actions[button])

        for _ in range(self.act_freq):
            self.pyboy.tick()

        # Release the button
        self.pyboy.send_input(self.release_button[button])

    def _get_state(self) -> np.ndarray:
        # Implement your state retrieval logic here
        game_stats = self._generate_game_stats()
        self.get_wall_status()
        is_grass = self._is_grass_tile()
        battle = (self._read_m(0xD057) != 0x00)
        state_vector = [
            game_stats["location"]["x"],
            game_stats["location"]["y"],
            game_stats["location"]["map_id"],
            # len(game_stats["pokemon"]),
            # sum(game_stats["levels"]),
            is_grass,
            battle,
            #self.enemy_hp,
            # game_stats["seen_pokemon"],
            # game_stats["caught_pokemon"],
            # sum(game_stats["xp"]),
            # self.read_hp_as_a_fraction(),
            # len(self.seen),
            # len(self.visited_coords),
            self.action,
            # obstacles?
            self.top_wall,
            self.bottom_wall,
            self.right_wall,
            self.left_wall
        ]

        return np.array(state_vector)

    def get_wall_status(self):
        map_data = self._get_screen_walkable_matrix()
        # Initialize direction flags
        walkable_left = 0
        walkable_right = 0
        walkable_up = 0
        walkable_down = 0
        player_x = 4
        player_y = 4

        # Check if the tile to the left is walkable
        if player_x > 0 and map_data[player_y][player_x - 1] == 1:
            walkable_left = 1

        # Check if the tile to the right is walkable
        if player_x < len(map_data[0]) - 1 and map_data[player_y][player_x + 1] == 1:
            walkable_right = 1

        # Check if the tile above is walkable
        if player_y > 0 and map_data[player_y - 1][player_x] == 1:
            walkable_up = 1

        # Check if the tile below is walkable
        if player_y < len(map_data) - 1 and map_data[player_y + 1][player_x] == 1:
            walkable_down = 1

        self.left_wall = walkable_left
        self.right_wall = walkable_right
        self.bottom_wall = walkable_down
        self.top_wall = walkable_up

    def penalty_walls(self):
        battle = (self._read_m(0xD057) != 0x00)
        penalty = 0
        if battle:
            return 0
        else:
            if self.left_wall == 0:
                penalty += -0.01
            if self.right_wall == 0:
                penalty += -0.01
            if self.top_wall == 0:
                penalty += -0.01
            if self.bottom_wall == 0:
                penalty += -0.01
        return penalty

    def read_hp_as_a_fraction(self) -> float:
        current_and_max_hp = self._read_party_hp()
        current_hp, max_hp = current_and_max_hp["current"], current_and_max_hp["max"]

        current_hp = sum(current_hp)
        max_hp = sum(max_hp)
        # make sure we don't divide by 0
        max_hp = max(max_hp, 1)

        return current_hp / max_hp

    enemy_hp = -1

    def update_enemy_hp(self) -> None:
        enemy_hp = self._read_m(0xCFE7)
        self.enemy_hp = enemy_hp

    def battle_reward(self):
        old_hp = self.enemy_hp
        self.update_enemy_hp()
        if self.enemy_hp < old_hp:
            print("TO BATTLE WE GO RAHHHHHHHHH")
            return 100
        return 0

    def _calculate_reward(self, new_state: dict) -> float:
        # Implement your reward calculation logic here
        temp_reward = 0
        temp_reward += -2
        battle_active = (self._read_m(0xD057) != 0x00)

        # check if new coordinate and reward.
        location = new_state.get("location")
        temp_reward += self.exploration_reward(location,battle_active)
        # check to ensure bro is not revisiting same spots
        # give a reward for if a pokemon is caught
        temp_reward += self._caught_reward(new_state) * 4

        # print(f"seen: {self._read_seen_pokemon_count()}")
        temp_reward += self._seen_reward(new_state) * 3

        # give a reward for battles indicated by pokemon level up
        temp_reward += self._levels_reward(new_state) * 5

        temp_reward += self.penalty_walls()

        if battle_active:
            #temp_reward += 2
            temp_reward += self.battle_reward()
        else:
            self.enemy_hp = -1

        if self._is_grass_tile():
            temp_reward += 10

        return temp_reward

    def exploration_reward(self, location, is_battle):
        key = f"{location}"
        reward = 0

        if key not in self.seen:
            self.seen.append(key)
        elif key in self.seen and not self._is_grass_tile() and not is_battle:
            return -1
        elif key in self.seen and self._is_grass_tile() and not is_battle:
            return -7

        if location["map_id"] not in self.visited_coords:
            self.visited_coords.append(location["map_id"])
            bruh = location["map_id"]
            print(f"new location!: {bruh}")
            if bruh == 12:
                return 500
            if (bruh != 40):
                return 100

        # if self.prior_game_stats["location"]["x"] != location["x"] or self.prior_game_stats["location"]["y"] != \
        #         location["y"]:
        #     reward += -1
        if self.prior_game_stats["location"]["y"] > location["y"]:
            reward += 3
        return reward

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        return game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here
        if self.steps >= 2000:
            self.visited_coords.clear()
            self.seen.clear()
            self.enemy_hp = -1
            self.action = -1
            return True
        return False
