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

        #======================================Left Over=====================================
        # game_stats["seen_pokemon"],
        # game_stats["caught_pokemon"],
        # sum(game_stats["xp"]),
        # self.read_hp_as_a_fraction(),
        # len(self.seen),
        # len(self.visited_coords),
        # ======================================Left Over=====================================

        if battle:  # logic being that when in battle the state vector should not matter, we want it to do a set action
            state_vector = [
                -1,
                -1,
                -1,
                battle,
                is_grass,
                self.read_enemy_hp_as_fraction(),  # normalise it
                self.action,

                # obstacles?
                -1,
                -1,
                -1,
                -1,
            ]
        else:  # traversing
            state_vector = [
                game_stats["location"]["x"],
                game_stats["location"]["y"],
                game_stats["location"]["map_id"],
                -1,
                -1,
                -1,
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

    def read_enemy_hp_as_fraction(self) -> float:
        enemy_max = self._read_m(0xCFF5)
        max_hp = max(enemy_max, 1)
        return self.enemy_hp / max_hp

    def read_hp_as_a_fraction(self) -> float:
        current_and_max_hp = self._read_party_hp()
        current_hp, max_hp = current_and_max_hp["current"], current_and_max_hp["max"]

        current_hp = sum(current_hp)
        max_hp = sum(max_hp)
        # make sure we don't divide by 0
        max_hp = max(max_hp, 1)

        return current_hp / max_hp
    prior_enemy_hp = -1
    enemy_hp = -1

    def update_enemy_hp(self) -> None:
        enemy_hp = self._read_m(0xCFE7)
        self.enemy_hp = enemy_hp

    def battle_reward(self):
        self.prior_enemy_hp = self.enemy_hp
        self.update_enemy_hp()
        if self.enemy_hp < self.prior_enemy_hp:
            print(f"TO BATTLE WE GO RAHHHHHHHHH: {self.prior_enemy_hp} - {self.enemy_hp} * 50 = {(self.prior_enemy_hp - self.enemy_hp) * 50}")
            return (self.prior_enemy_hp - self.enemy_hp) * 50
        return 0

    def has_won(self, new_state):
        if self.enemy_hp == 0 and self.prior_enemy_hp !=0 :
            joe = 100 * (sum(new_state["levels"]) - sum(self.prior_game_stats["levels"]))
            print(f"KILLED AN ENEMY: {joe}")
            enemy_max = self._read_m(0xCFF5)
            return 100 * enemy_max
        return 0

    def _calculate_reward(self, new_state: dict) -> float:
        # Implement your reward calculation logic here
        temp_reward = 0
        battle_active = (self._read_m(0xD057) != 0x00)

        # exploration rewards:
        location = new_state.get("location")
        temp_reward += self.exploration_reward(location, battle_active)
        temp_reward += self.penalty_walls()

        # battle rewards
        if battle_active:
            temp_reward += -7  # penalise idle battle state
            temp_reward += self.battle_reward()
        else:
            self.enemy_hp = -1
            self.prior_enemy_hp = -1
        temp_reward += self.has_won(new_state)

        return temp_reward

    def exploration_reward(self, location, battle_active):
        key = f"{location}"
        reward = 0

        # if unseen then reward it and add it
        if key not in self.seen:
            self.seen.append(key)
            reward+=5

        if self._is_grass_tile():
            reward += 7
        #
        if key in self.seen and self._is_grass_tile() and not battle_active:  # only penalise staying in the same place if its grass and not battle
            reward += -1

        if location["map_id"] not in self.visited_coords:
            self.visited_coords.append(location["map_id"])
            bruh = location["map_id"]
            print(f"new location!: {bruh}")
            if (bruh != 40):
                return 1000

        # Remove is not grass and the function below it, as it was getting to killin things before this.
        if self.prior_game_stats["location"]["x"] == location["x"] and self.prior_game_stats["location"]["y"] == \
                location["y"] and not battle_active:
            reward += -2

        # if self.prior_game_stats["location"]["x"] == location["x"] and self.prior_game_stats["location"]["y"] == \
        #         location["y"] and not battle_active and self._is_grass_tile():
        #     reward += -6

        if self.prior_game_stats["location"]["x"] != location["x"] or self.prior_game_stats["location"]["y"] != \
                location["y"]:
            reward += -1

        if self.prior_game_stats["location"]["y"] > location["y"] and self.prior_game_stats["location"]["map_id"] == \
                location["map_id"] and location["map_id"] ==12:
            reward += 5 * (1/(abs(0-location["y"])+1))
            print(f"reward is: {reward}")
            y = self.prior_game_stats["location"]["y"]
            x = self.prior_game_stats["location"]["x"]
            print(f"prior location: {x},{y}")
            print(f"current:{location}")

        elif self.prior_game_stats["location"]["y"] > location["y"] and self.prior_game_stats["location"]["map_id"] == \
                location["map_id"]:
            reward+=2 * (1/(abs(0-location["y"])+1))
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
            self.prior_enemy_hp=-1
            self.action = -1
            self.left_wall = -1
            self.right_wall = -1
            self.top_wall = -1
            self.bottom_wall = -1
            return True
        return False
